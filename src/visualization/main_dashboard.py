import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
from streamlit_folium import st_folium
import datetime
import dataset_explorer
import graphs


st.set_page_config(page_title="Resource Allocation Forecaster", layout="wide")

@st.cache_data
def load_data():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    police_path = data_dir / "SHP" / "Police_Force_Areas_UK.shp"
    lsoa_path = data_dir / "SHP" / "LSOA" / "LSOA.shp"
    parquet_path = data_dir / "master_dataset_full_set_no_solved_percent.parquet"
    oco_path = data_dir / "oco_outlier_optimised.parquet"
    sarima_path = data_dir / "SARIMA_forecast.csv"

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
        return None, None, None, None, None

    try:
        database = pd.read_parquet(parquet_path)
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None, None, None, None
    
    try:
        oco_prediction = pd.read_parquet(oco_path)
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None, None, None, None
    
    try:
        sarima_prediction = pd.read_csv(sarima_path)
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None, None, None, None

    return police_gdf, lsoa_gdf, database, oco_prediction, sarima_prediction

police_gdf, lsoa_gdf, df, oco_df, sarima_df = load_data()
# Rename empty column names
sarima_df.rename(columns={sarima_df.columns[0]: 'Index', sarima_df.columns[1]: 'LSOA code', sarima_df.columns[2]: 'Predicted'}, inplace=True)


# Session state/ memory
if 'clicked_force' not in st.session_state:
    st.session_state.clicked_force = None

if 'clicked_lsoa' not in st.session_state:
    st.session_state.clicked_lsoa = None

# Headers and filters
def head_and_filt():
    st.title("Resource Allocation & Demand Forecaster")
    this_month = datetime.date.today()
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        # target_month = st.selectbox("Target Month", ["August 2026", "September 2026"])
        st.markdown("Target Month:")
        st.subheader(str((this_month + datetime.timedelta(days=32)).strftime("%B %Y")))
    with filter_col2:
        if df is None:
            st.error("Failed to load the main dataset. Check that the data files exist and the paths are correct.")
            st.stop()
        # crime_type = st.selectbox("Crime Focus", df['Crime type'].dropna().unique().tolist())


# Main layout
def main_layout():
    col1, col2 = st.columns([2, 1])

    if police_gdf is not None:
        with col1:
            # national map
            
            st.subheader("Select a Police Force to evaluate")
            
            m = folium.Map(
                location=[52.5, 0.5],
                zoom_start=6,
                min_zoom=6,          # Prevents zooming out further than the default view
                max_bounds=True,     # Activates the panning boundaries
                min_lat=49.5,        # Southernmost point (approx)
                max_lat=61.0,        # Northernmost point (approx)
                min_lon=-8.0,        # Westernmost point (approx)
                max_lon=3.5          # Easternmost point (approx)
            )
            
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
                st.info(clicked_name)
                st.rerun(scope='fragment')
                
            
        with col2:
            st.info("Please click a Police Force on the map to begin the analysis.")


# zoomed LSOAs fragment
def zoomed_lsoa():
    # build base force dataframe
    base_force_df = df[df['Police Territory'].str.contains(st.session_state.clicked_force)].copy()

    force_df = base_force_df.merge(
        sarima_df,
        on='LSOA code'
    )
    # Controls for highlighting
    highlight_hotspots = st.checkbox("Highlight predicted hotspots", value=True)
    try:
        pred_max = float(force_df['Predicted'].dropna().max())
    except Exception:
        pred_max = 100.0
    threshold = st.slider("Prediction threshold", 0.0, max(pred_max, 1.0), float(min(10.0, pred_max)))

    # Compute set of LSOA codes to highlight (based on current police force)
    highlight_codes = set()
    if highlight_hotspots and st.session_state.clicked_force:
        try:
            force_pred = df[df['Police Territory'].str.contains(st.session_state.clicked_force)].merge(
                sarima_df,
                on='Predicted'
            )
            highlight_codes = set(force_pred[force_pred['Predicted'].astype(float) > float(threshold)]['LSOA code'].astype(str).tolist())
        except Exception:
            highlight_codes = set()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Localized LSOA Forecast: {st.session_state.clicked_force}")
                    
        # button to go back
        if st.button("⬅️ Back to National Map"):
            st.session_state.clicked_force = None
            st.rerun(scope='fragment')
            
            
        # region masking
        force_geom = police_gdf[police_gdf['PFANM'] == st.session_state.clicked_force]
        clipped_lsoas = gpd.clip(lsoa_gdf, force_geom)
        
        center_lat = force_geom.geometry.centroid.y.iloc[0]
        center_lon = force_geom.geometry.centroid.x.iloc[0]
        
        m2 = folium.Map(location=[center_lat, center_lon], zoom_start=9, min_zoom=8)
        
        # draw the thick Police boundary over it
        folium.GeoJson(
            force_geom,
            style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 4}
        ).add_to(m2)

        # draw the clipped LSOAs with dynamic styling based on predictions
        def lsoa_style(feature):
            props = feature.get('properties', {}) or {}
            # possible property keys that contain the LSOA code
            code = None
            for k in ('LSOA21CD', 'LSOA11CD', 'LSOA Code', 'LSOA_code', 'LSOA code'):
                if k in props:
                    code = props[k]
                    break
            # highlight if code in computed set
            try:
                if code is not None and str(code) in highlight_codes:
                    return {'fillColor': '#ff5733', 'color': 'red', 'weight': 1, 'fillOpacity': 0.6}
            except Exception:
                pass
            return {'fillColor': 'transparent', 'color': 'red', 'weight': 1}

        folium.GeoJson(
            clipped_lsoas,
            tooltip=folium.GeoJsonTooltip(fields=['LSOA21NM']),
            style_function=lsoa_style
        ).add_to(m2)
        
        map_data_lsoa = st_folium(m2, height=500, use_container_width=True, key="zoomed_map")

        if map_data_lsoa and map_data_lsoa.get('last_active_drawing'):
            clicked_properties = map_data_lsoa['last_active_drawing']['properties']
            if 'LSOA21NM' in clicked_properties:
                clicked_lsoa_name = clicked_properties['LSOA21NM']
                st.session_state.clicked_lsoa = clicked_lsoa_name
                st.toast(f"Selected: {clicked_lsoa_name}")
                st.rerun(scope='fragment')

    with col2: 
            if st.session_state.clicked_force:
                st.subheader("Seasonal Comparison")
                # display_df = df[['Name', 'Seasonal_Baseline', 'Predicted_Count', 'Predicted_Spike']].sort_values(by='Predicted_Spike', ascending=False)
                display_df = force_df[['LSOA code','LSOA name', 'Predicted']].sort_values('Predicted', ascending=False).head()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.subheader("Resource Recommendation")
                selected_lsoa = st.selectbox("Select Target Area:", force_df['LSOA name'])

                lsoa_data = force_df[force_df['LSOA name'] == selected_lsoa].iloc[0]
                #TODO: include predictions
            #     spike = lsoa_data['Predicted_Spike']
            #     is_localized = lsoa_data['Is_Localized']

            #     st.metric(label=f"Spike vs. {target_month.split()[0]} Baseline", value=f"+{spike}%")

            #     if spike > 15.0 and is_localized:
            #         st.warning("⚠️ **Spatial Analysis: Localized Spike Detected.**\n\n**Recommendation:** Temporarily redistribute flexible patrol time from low-risk zones to this hotspot. Do not permanently increase overall force headcount.")
            #     elif spike > 15.0 and not is_localized:
            #         st.info("📊 **Spatial Analysis: Broad Increase Detected.**\n\n**Recommendation:** The predicted increase is spread across the majority of the police force area. A targeted hotspot patrol response is not recommended here.")
            #     else:
            #         st.success("✅ **Spatial Analysis: Normal Baseline.**\n\n**Recommendation:** Expected crime levels are within standard seasonal variations. Maintain normal allocations.")
            # else:
            #     st.info("Please click a Police Force on the map to begin the analysis.")

@st.fragment
# switches the different map layouts
def dynamic_fragment_container():
    if st.session_state.clicked_force is None:
        main_layout()
        
    else:
        zoomed_lsoa()

def tab1():
    # Layout configuration
    head_and_filt()
    st.markdown("---")
    dynamic_fragment_container()
    st.toast("Done loading")


# Streamlit doesn't allow more than 1 lambda function :(
def explorer_page():
    return dataset_explorer.crime_dataset_explorer(df)

def graphs_page():
    return graphs.tab3(df)

# Navigation menu for pages
pg = st.navigation([
    st.Page(tab1, title="Map", default=True),
    st.Page(explorer_page, title="Explorer"),
    st.Page(graphs_page, title="Graphs")
], position="top")

pg.run()