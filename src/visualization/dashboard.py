import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
from streamlit_folium import st_folium
import datetime
from dateutil import relativedelta

st.set_page_config(page_title="Resource Allocation Forecaster", layout="wide")

#TODO:
# remove Scotland and Ireland
# replace dummy database with current one
# make LSOAs clickable
# interconnect data with selectboxes
# graphs (optional currently)

# Loading of database
@st.cache_data
def load_data():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    police_path = data_dir / "SHP" / "Police_Force_Areas_UK.shp"
    lsoa_path = data_dir / "SHP" / "LSOA" / "LSOA.shp"
    parquet_path = data_dir / "master_dataset_full_set_no_solved_percent.parquet"

    try:
        police_gdf = gpd.read_file(police_path).to_crs(epsg=4326)
        exclude_names = [
            "Scotland",
            "Northern Ireland"
        ]
        police_gdf = police_gdf[~police_gdf['PFANM'].isin(exclude_names)].copy()
        lsoa_gdf = gpd.read_file(lsoa_path).to_crs(epsg=4326)

        police_gdf['geometry'] = police_gdf.geometry.simplify(0.005)
        lsoa_gdf['geometry'] = lsoa_gdf.geometry.simplify(0.005)
    except Exception as e:
        st.error(f"Shapefile error: {e}")
        return None, None, None

    try:
        database = pd.read_parquet(parquet_path)
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None, None

    return police_gdf, lsoa_gdf, database



police_gdf, lsoa_gdf, df = load_data()

# session state/ memory
if 'clicked_force' not in st.session_state:
    st.session_state.clicked_force = None

# headers and filters
st.title("Resource Allocation & Demand Forecaster")
this_month = datetime.date.today()
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    # target_month = st.selectbox("Target Month", ["August 2026", "September 2026"])
    target_month = st.selectbox("Target Month", [(this_month + datetime.timedelta(days=32)).strftime("%B %Y"), (this_month + datetime.timedelta(days=64)).strftime("%B %Y")])
with filter_col2:
    if df is None:
        st.error("Failed to load the main dataset. Check that the data files exist and the paths are correct.")
        st.stop()
    crime_type = st.selectbox("Crime Focus", df['Crime type'].dropna().unique().tolist())

st.markdown("---")

# main layout
col1, col2 = st.columns([2, 1])

if police_gdf is not None:
    with col1:
        # national map
        if st.session_state.clicked_force is None:
            st.subheader("Select a Police Force to evaluate")
            
            m = folium.Map(location=[52.5, -1.5], zoom_start=6)
            
            # draw polygons and make them clickable
            folium.GeoJson(
                police_gdf,
                name="Police Forces",
                # NOTE: Check your shapefile! If the column isn't 'PFANM', change this to match your data
                tooltip=folium.GeoJsonTooltip(fields=['PFANM']),
                style_function=lambda x: {'fillColor': '#3186cc', 'color': 'black', 'weight': 1, 'fillOpacity': 0.4}
            ).add_to(m)

            # 
            map_data = st_folium(m, height=500, use_container_width=True, key="national_map")

            # TODO: fix dark screen when interacting with the map
            if map_data and map_data.get('last_active_drawing'):
                clicked_name = map_data['last_active_drawing']['properties']['PFANM']
                st.session_state.clicked_force = clicked_name
                st.rerun() 

        # zoomed lsoa
        else:
            st.subheader(f"Localized LSOA Forecast: {st.session_state.clicked_force}")
            
            # button to go back
            if st.button("⬅️ Back to National Map"):
                st.session_state.clicked_force = None
                st.rerun()
                
            # region masking
            force_geom = police_gdf[police_gdf['PFANM'] == st.session_state.clicked_force]
            clipped_lsoas = gpd.clip(lsoa_gdf, force_geom)
            
            center_lat = force_geom.geometry.centroid.y.iloc[0]
            center_lon = force_geom.geometry.centroid.x.iloc[0]
            
            m2 = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            # draw the clipped LSOAs
            folium.GeoJson(
                clipped_lsoas,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 1}
            ).add_to(m2)
            
            # draw the thick Police boundary over it
            folium.GeoJson(
                force_geom,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 4}
            ).add_to(m2)
            
            st_folium(m2, height=500, use_container_width=True, key="zoomed_map")

    # sidebar data that shows only when LSOA is clicked
    with col2:
        if st.session_state.clicked_force:
            st.subheader("Seasonal Comparison")
            display_df = df[['Name', 'Seasonal_Baseline', 'Predicted_Count', 'Predicted_Spike']].sort_values(by='Predicted_Spike', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.subheader("Resource Recommendation")
            selected_lsoa = st.selectbox("Select Target Area:", df['Name'])

            lsoa_data = df[df['Name'] == selected_lsoa].iloc[0]
            spike = lsoa_data['Predicted_Spike']
            is_localized = lsoa_data['Is_Localized']

            st.metric(label=f"Spike vs. {target_month.split()[0]} Baseline", value=f"+{spike}%")

            if spike > 15.0 and is_localized:
                st.warning("⚠️ **Spatial Analysis: Localized Spike Detected.**\n\n**Recommendation:** Temporarily redistribute flexible patrol time from low-risk zones to this hotspot. Do not permanently increase overall force headcount.")
            elif spike > 15.0 and not is_localized:
                st.info("📊 **Spatial Analysis: Broad Increase Detected.**\n\n**Recommendation:** The predicted increase is spread across the majority of the police force area. A targeted hotspot patrol response is not recommended here.")
            else:
                st.success("✅ **Spatial Analysis: Normal Baseline.**\n\n**Recommendation:** Expected crime levels are within standard seasonal variations. Maintain normal allocations.")
        else:
            st.info("Please click a Police Force on the map to begin the analysis.")