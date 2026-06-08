import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
from streamlit_folium import st_folium
import datetime
from dateutil import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Resource Allocation Forecaster", layout="wide")

# Create tabs for each components
tab1, tab2, tab3 = st.tabs(["Map", "Data Explorer", "Graphs"])

# Add Resource Allocation Forecaster to the first tab
with tab1:
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

    # Session state/ memory
    if 'clicked_force' not in st.session_state:
        st.session_state.clicked_force = None

    if 'clicked_lsoa' not in st.session_state:
        st.session_state.clicked_lsoa = None

    # Headers and filters
    @st.fragment(run_every=None)
    def head_and_filt():
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

            #draw the clipped LSOAs
            folium.GeoJson(
                clipped_lsoas,
                tooltip=folium.GeoJsonTooltip(fields=['LSOA21NM']),
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 1}
            ).add_to(m2)
            
            # draw the thick Police boundary over it
            folium.GeoJson(
                force_geom,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 4}
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
                    force_df = df[df['Police Territory'].str.contains(st.session_state.clicked_force)]
                    st.subheader("Seasonal Comparison")
                    # display_df = df[['Name', 'Seasonal_Baseline', 'Predicted_Count', 'Predicted_Spike']].sort_values(by='Predicted_Spike', ascending=False)
                    display_df = force_df[['LSOA code','LSOA name', 'Crime type']].head()
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    st.subheader("Resource Recommendation")
                    selected_lsoa = st.selectbox("Select Target Area:", force_df['LSOA name'])

                    lsoa_data = force_df[force_df['LSOA name'] == selected_lsoa].iloc[0]
                    #TODO: include predictions
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

    @st.fragment
    # switches the different map layouts
    def dynamic_fragment_container():
        if st.session_state.clicked_force is None:
            main_layout()
        else:
            zoomed_lsoa()

    # Layout configuration
    head_and_filt()
    st.markdown("---")
    dynamic_fragment_container()

#Add Crime Dataset Explorer to the second tab
with tab2: 
    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
    
    col1, col2 = st.columns([1, 4])
    
    col2.title("Crime Dataset Explorer")

    DATA_PATH = "data/master_dataset_dropped_outcome_nulls.parquet"
    df = load_data(DATA_PATH)

    col1.header("Filter dataset")

    month_options = sorted(df["Month"].dropna().unique())
    selected_months = col1.multiselect("Month", month_options, default=month_options[:5])

    territory_options = sorted(df["Police Territory"].dropna().unique())
    with col1:
        with st.expander("Police Territory"):
            select_all_territories = st.checkbox("Select all territories", value=True)
            selected_territories = []
            for territory in territory_options:
                checked = st.checkbox(territory, value=select_all_territories)
                if checked:
                    selected_territories.append(territory)
                

    crime_type_options = sorted(df["Crime type"].dropna().unique())
    # selected_crime_types = col1.multiselect("Crime type", crime_type_options, default=crime_type_options)
    with col1:
        with st.expander("Crime Type"):
            select_all_crimes = st.checkbox("Select all crime types", value=True)
            selected_crime_types = []
            for crime in crime_type_options:
                checked = st.checkbox(crime, value=select_all_crimes)
                if checked:
                    selected_crime_types.append(crime)

    rural_urban_options = sorted(df["Rural_Urban"].dropna().unique())
    selected_rural_urban = col1.multiselect("Rural / Urban", rural_urban_options, default=rural_urban_options)

    col1.markdown("---")
    col1.subheader("Numeric filters")

    crime_count_min, crime_count_max = int(df["Crime_Count"].min()), int(df["Crime_Count"].max())
    crime_count_range = col1.slider("Crime Count", crime_count_min, crime_count_max, (crime_count_min, min(crime_count_min + 20, crime_count_max)))

    solved_min, solved_max = float(df["Solved_%"].min()), float(df["Solved_%"].max())
    solved_range = col1.slider("Solved %", solved_min, solved_max, (solved_min, solved_max), format="%.1f")

    temp_min, temp_max = float(df["Mean_Temp"].min()), float(df["Mean_Temp"].max())
    temp_range = col1.slider("Mean Temperature", temp_min, temp_max, (temp_min, temp_max), format="%.1f")

    sun_min, sun_max = float(df["Sunshine_Duration"].min()), float(df["Sunshine_Duration"].max())
    sun_range = col1.slider("Sunshine Duration", sun_min, sun_max, (sun_min, sun_max), format="%.1f")

    rain_min, rain_max = float(df["Total_Rainfall"].min()), float(df["Total_Rainfall"].max())
    rain_range = col1.slider("Total Rainfall", rain_min, rain_max, (rain_min, rain_max), format="%.1f")

    if not selected_months:
        selected_months = month_options
    if not selected_territories:
        selected_territories = territory_options
    if not selected_crime_types:
        selected_crime_types = crime_type_options
    if not selected_rural_urban:
        selected_rural_urban = rural_urban_options

    filtered_df = df[
        df["Month"].isin(selected_months)
        & df["Police Territory"].isin(selected_territories)
        & df["Crime type"].isin(selected_crime_types)
        & df["Rural_Urban"].isin(selected_rural_urban)
        & df["Crime_Count"].between(crime_count_range[0], crime_count_range[1])
        & df["Solved_%"].between(solved_range[0], solved_range[1])
        & df["Mean_Temp"].between(temp_range[0], temp_range[1])
        & df["Sunshine_Duration"].between(sun_range[0], sun_range[1])
        & df["Total_Rainfall"].between(rain_range[0], rain_range[1])
    ]

    col2.markdown("### Filtered dataset ")

    # selected_columns = st.multiselect(
    #     "Columns to display",
    #     options=df.columns.tolist(),
    #     default=df.columns.tolist(),
    # )

    col2.dataframe(filtered_df, hide_index=True)

#Create the third tab for generating graphs
with tab3:
    st.header("Graphs")
    st.title("Crime Data Graph")

    graph_df = df.copy()
    
    # Split data in years
    graph_df["Year"] = graph_df["Month"].astype(str).str[:4]

    months = {
        "01" : "Jan",
        "02" : "Feb",
        "03" : "Mar",
        "04" : "Apr",
        "05" : "May",
        "06" : "June",
        "07" : "July",
        "08" : "Aug",
        "09" : "Sep",
        "10" : "Oct",
        "11" : "Nov",
        "12" : "Dec"
    }

    # Split data in months
    graph_df["Month"] = graph_df["Month"].astype(str).str[5:].replace(months)

    # Use the name of the months instead of numbers with dictionary
    graph_df["Month"] = pd.Categorical(
        graph_df["Month"], 
        categories=list(months.values()), 
        ordered=True
    )

    # Add columns instead of sidebars to prevent sidebar being visible in every tab.
    col1, col2 = st.columns(2)

    # Selectbox for years
    with col1:
        years = sorted(graph_df["Year"].unique())
        chosen_year = st.selectbox("Choose Year:", options=["All"] + years)

    #Multiselect box for crimes
    with col2:
        crimes = sorted(graph_df['Crime type'].unique())
        selected_crimes = st.multiselect(
            "Select Crime Type(s):", 
            options=crimes, 
            default=[crimes[0]] 
        )
    
    if selected_crimes:
        graph_df = graph_df[graph_df['Crime type'].isin(selected_crimes)]


    if not graph_df.empty:
        
        #If no year is chosen, show all years crime count
        if chosen_year == "All":
            grouped_df = graph_df.groupby(['Year', 'Crime type'])['Crime_Count'].sum().reset_index()
            x_column = "Year"
            x_label = "Year"
            chart_title = "Crime Count by Year"
        
        #If a specfic year is chosen, show monthly crime counts 
        else:
            graph_df = graph_df[graph_df['Year'] == chosen_year]  
            grouped_df = graph_df.groupby(['Month', 'Crime type'])['Crime_Count'].sum().reset_index()
            x_column = "Month"
            x_label = "Month"
            chart_title = f"Crime Count by Month in {chosen_year}"

        # Create figure for plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add seaborn bar plot
        sns.barplot(
            data=grouped_df, 
            x=x_column,      # choose month or year according to chosen year 
            y="Crime_Count", 
            hue="Crime type", 
            ax=ax,
            palette="viridis" 
        )
        # Add labels and titles to the graph
        ax.set_title(chart_title, fontsize=16, pad=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Total Crimes", fontsize=12)
        
        # Add legend and move to upper left
        plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Show the plot in dashboard
        st.pyplot(fig)