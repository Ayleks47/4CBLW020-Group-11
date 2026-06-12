import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
from streamlit_folium import st_folium
import datetime
import dataset_explorer
import graphs
import pulp
import json

st.set_page_config(page_title="Resource Allocation Forecaster", layout="wide")

# 1. OPTIMIZED DATA LOADING
@st.cache_resource
def load_and_prep_spatial_models():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    
    # 1. LOAD GEOMETRIES & ESTABLISH GEOGRAPHIC TRUTH
    police_gdf = gpd.read_file(data_dir / "SHP" / "Police_Force_Areas_UK.shp").to_crs(epsg=4326)
    police_gdf = police_gdf[~police_gdf['PFANM'].isin(["Scotland", "Northern Ireland"])].copy()
    
    lsoa_gdf = gpd.read_file(data_dir / "SHP" / "LSOA" / "LSOA.shp").to_crs(epsg=4326)
    lsoa_centroids = lsoa_gdf.copy()
    lsoa_centroids['geometry'] = lsoa_centroids.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    
    joined = gpd.sjoin(lsoa_centroids, police_gdf[['PFANM', 'geometry']], how='inner', predicate='within')
    lsoa_gdf['PFANM'] = joined['PFANM']
    
    police_gdf['geometry'] = police_gdf.geometry.simplify(0.01)
    lsoa_gdf['geometry'] = lsoa_gdf.geometry.simplify(0.015)

    geo_truth = lsoa_gdf[['LSOA21CD', 'LSOA21NM', 'PFANM']].copy()
    geo_truth.columns = ['LSOA code', 'LSOA name', 'Police_Force_Map']

    # 2. LOAD & FLATTEN OCO
    try:
        oco = pd.read_parquet(data_dir / "oco_outlier_optimised.parquet")
        oco_baseline = oco.groupby('LSOA_code')['total_crimes'].median().reset_index()
        oco_baseline.columns = ['LSOA code', 'Baseline']
        oco_baseline = oco_baseline.drop_duplicates(subset=['LSOA code'])
        
        latest_oco_month = oco['Month'].max()
        oco_latest = oco[oco['Month'] == latest_oco_month][['LSOA_code', 'predicted_crime']]
        oco_latest.columns = ['LSOA code', 'Predicted_OCO']
        oco_latest = oco_latest.drop_duplicates(subset=['LSOA code'])
    except Exception:
        oco_baseline = pd.DataFrame(columns=['LSOA code', 'Baseline'])
        oco_latest = pd.DataFrame(columns=['LSOA code', 'Predicted_OCO'])

    # 3. LOAD & FLATTEN SARIMA
    try:
        sarima = pd.read_csv(data_dir / "SARIMA_forecast.csv")
        sarima_latest = sarima[['LSOA Code', 'Unnamed: 2']].rename(columns={'LSOA Code': 'LSOA code', 'Unnamed: 2': 'Predicted_SARIMA'})
        sarima_latest = sarima_latest.drop_duplicates(subset=['LSOA code'])
    except Exception:
        sarima_latest = pd.DataFrame(columns=['LSOA code', 'Predicted_SARIMA'])

    # 4. BUILD THE MASTER MILP DATAFRAME
    geo_truth = geo_truth.drop_duplicates(subset=['LSOA code'])
    
    master_milp = geo_truth.merge(oco_baseline, on='LSOA code', how='left')
    master_milp = master_milp.merge(oco_latest, on='LSOA code', how='left')
    master_milp = master_milp.merge(sarima_latest, on='LSOA code', how='left')
    
    numeric_cols = ['Baseline', 'Predicted_OCO', 'Predicted_SARIMA']
    master_milp[numeric_cols] = master_milp[numeric_cols].fillna(0)
    master_milp = master_milp.reset_index(drop=True).copy()

    return police_gdf, lsoa_gdf, master_milp

@st.cache_resource
def load_database():
    try:
        return pd.read_parquet(Path(__file__).resolve().parents[2] / "data" / "master_dataset_full_set_no_solved_percent.parquet")
    except Exception:
        return None

@st.cache_resource
def load_adjacency_matrix():
    try:
        with open(Path(__file__).resolve().parents[2] / "data" / "lsoa_neighbors.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}

# Cleaned up global instantiations (no dead functions)
police_gdf, lsoa_gdf, master_milp_df = load_and_prep_spatial_models()
df = load_database()
neighbors_dict = load_adjacency_matrix()

if 'clicked_force' not in st.session_state: st.session_state.clicked_force = None
if 'clicked_lsoa' not in st.session_state: st.session_state.clicked_lsoa = None


# 2. MILP OPTIMIZATION ENGINE
def run_milp_optimization(opt_df, total_hours, beta, c_max):
    epsilon = 0.01
    weights = {}
    sum_baseline = opt_df['Baseline'].sum() if opt_df['Baseline'].sum() > 0 else 0.01
        
    for _, row in opt_df.iterrows():
        F_i, B_i, lsoa = row['Predicted'], row['Baseline'], row['LSOA code']
        weights[lsoa] = (F_i - B_i) / (B_i + epsilon) if F_i > 10 else 0.0
            
    prob = pulp.LpProblem("Patrol_Allocation", pulp.LpMaximize)
    lsoas = opt_df['LSOA code'].unique().tolist()
    
    x_vars = pulp.LpVariable.dicts("Hours", lsoas, lowBound=0, cat='Continuous')
    prob += pulp.lpSum([weights[i] * x_vars[i] for i in lsoas]), "Objective"
    prob += pulp.lpSum([x_vars[i] for i in lsoas]) <= total_hours, "Capacity_Constraint"
    
    infeasible_risk = False
    for i in lsoas:
        B_i = opt_df.loc[opt_df['LSOA code'] == i, 'Baseline'].values[0]
        min_hours = beta * (total_hours * (B_i / sum_baseline))
        max_hours = c_max * total_hours
        
        if min_hours > max_hours: infeasible_risk = True
        prob += x_vars[i] >= min_hours, f"Floor_{i}"
        prob += x_vars[i] <= max_hours, f"Cap_{i}"
        
    if infeasible_risk: return "Infeasible", None
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    results = []
    for i in lsoas:
        results.append({
            'LSOA code': i,
            'LSOA name': opt_df.loc[opt_df['LSOA code'] == i, 'LSOA name'].values[0],
            'Forecast': round(opt_df.loc[opt_df['LSOA code'] == i, 'Predicted'].values[0], 1),
            'Surge Weight': round(weights[i], 3),
            'Assigned Hours': round(x_vars[i].varValue, 1)
        })
    return pulp.LpStatus[prob.status], pd.DataFrame(results).sort_values(by='Assigned Hours', ascending=False)

# 3. UI SIDEBAR & CONTIGUITY ANALYSIS
@st.fragment
def milp_ui_sidebar(force_df):
    st.subheader("Operational Constraints (MILP)")
    
    total_hours = st.number_input("Total Force Capacity (Hours)", min_value=100, value=5000, step=100)
    beta_protection = st.slider("Baseline Protection Factor", 0.0, 1.0, 0.5)
    max_cap = st.slider("Maximum Saturation Cap", 0.01, 1.0, 0.10)
    
    if st.button("Generate Optimal Deployment Schedule", type="primary"):
        with st.spinner("Optimizing deployment..."):
            status, optimal_df = run_milp_optimization(force_df, total_hours, beta_protection, max_cap)
            
            if status == "Infeasible":
                st.error("❌ **Constraint Error:** Your Maximum Cap is too restrictive to support your Baseline Protection. Please adjust the sliders.")
            elif status == "Optimal":
                st.success("✅ **Optimal Schedule Generated**")
                st.dataframe(optimal_df[['LSOA name', 'Forecast', 'Assigned Hours']].head(10), width='stretch', hide_index=True)
            else:
                st.warning(f"Solver Status: {status}")

    st.markdown("---")
    
    if st.session_state.clicked_lsoa:
        st.subheader(f"Area Details: {st.session_state.clicked_lsoa}")
        if st.button("✕ Clear Selection"):
            st.session_state.clicked_lsoa = None
            st.rerun(scope='fragment')
        
        try:
            # FIX: Simplified display logic for flat data
            display_df = force_df[force_df['LSOA name'] == st.session_state.clicked_lsoa][['LSOA name', 'Predicted']].copy()
            display_df['Predicted'] = display_df['Predicted'].round(1)
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # FIX: Streamlined 50% Contiguity Rule Logic
            st.markdown("### Spatial Contiguity Analysis")
            adjacent_lsoas = neighbors_dict.get(st.session_state.clicked_lsoa, [])
            
            if adjacent_lsoas:
                total_neighbors, surging_neighbors = 0, 0
                for neighbor in adjacent_lsoas:
                    # force_df is already perfectly filtered to the current map state and active model
                    neighbor_data = force_df[force_df['LSOA name'] == neighbor]
                    if not neighbor_data.empty:
                        total_neighbors += 1
                        if neighbor_data['Predicted'].sum() > 10: # Surge threshold
                            surging_neighbors += 1
                
                if total_neighbors > 0:
                    surge_ratio = surging_neighbors / total_neighbors
                    st.progress(surge_ratio)
                    if surge_ratio >= 0.5:
                        st.error(f"🚨 **Macro-Surge Detected**\n\n{surging_neighbors} of {total_neighbors} immediate adjacent neighborhoods are also forecasting surges. **Do not pull resources from neighbors.** Rely entirely on the force-wide MILP redistribution.")
                    else:
                        st.warning(f"⚠️ **Micro-Spike Detected**\n\nOnly {surging_neighbors} of {total_neighbors} immediate adjacent neighborhoods are surging. You may safely shift discretionary patrol time from quiet adjacent zones.")
            else:
                st.info("No spatial adjacency data available for this LSOA.")
                
        except Exception as e:
            st.warning(f"No prediction data available for this specific LSOA. Error: {e}")
    else:
        st.info("Click an LSOA on the map to view spatial analysis and context.")

# 4. MAP VIEWS & STREAMLIT ROUTING
def head_and_filt():
    st.title("Resource Allocation & Demand Forecaster")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        st.markdown("Target Month:")
        st.subheader(str((datetime.date.today() + datetime.timedelta(days=32)).strftime("%B %Y")))
    with filter_col2:
        if df is None: st.error("Failed to load dataset."); st.stop()

def main_layout():
    col1, col2 = st.columns([2, 1])
    if police_gdf is not None:
        with col1:
            st.subheader("Select a Police Force to evaluate")
            m = folium.Map(location=[52.5, 0.5], zoom_start=6, min_zoom=6, max_bounds=True, min_lat=49.5, max_lat=61.0, min_lon=-8.0, max_lon=3.5)
            folium.GeoJson(police_gdf, tooltip=folium.GeoJsonTooltip(fields=['PFANM']), style_function=lambda x: {'fillColor': '#3186cc', 'color': 'black', 'weight': 1, 'fillOpacity': 0.4}).add_to(m)
            map_data = st_folium(m, height=500, width='stretch', key="national_map", returned_objects=["last_active_drawing"])

            if map_data and map_data.get('last_active_drawing'):
                st.session_state.clicked_force = map_data['last_active_drawing']['properties']['PFANM']
                st.rerun(scope='fragment')
        with col2: st.info("Please click a Police Force on the map to begin the analysis.")

def zoomed_lsoa():
    force_geom = police_gdf[police_gdf['PFANM'] == st.session_state.clicked_force]
    clipped_lsoas = lsoa_gdf[lsoa_gdf['PFANM'] == st.session_state.clicked_force][['LSOA21NM', 'geometry']]
    
    force_milp_data = master_milp_df[master_milp_df['Police_Force_Map'] == st.session_state.clicked_force].copy()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Forecast Engine")
        model_choice = st.radio("Select Prediction Model:", ["SARIMA", "OCO (Optimised)"], horizontal=True)
        
        if model_choice == "SARIMA":
            force_milp_data['Predicted'] = force_milp_data['Predicted_SARIMA']
        else:
            force_milp_data['Predicted'] = force_milp_data['Predicted_OCO']
            
        force_milp_data = force_milp_data[~force_milp_data['LSOA name'].str.contains('Unknown|None', na=False, case=False)]

    with col1:
        st.subheader(f"Localized LSOA Forecast: {st.session_state.clicked_force}")
        if st.button("⬅️ Back to National Map"):
            st.session_state.clicked_force = None
            st.session_state.clicked_lsoa = None
            st.rerun(scope='fragment')
            
        projected_geom = force_geom.to_crs(epsg=3857)
        center_lat = projected_geom.geometry.centroid.to_crs(epsg=4326).y.iloc[0]
        center_lon = projected_geom.geometry.centroid.to_crs(epsg=4326).x.iloc[0]
        
        m2 = folium.Map(location=[center_lat, center_lon], zoom_start=9, min_zoom=8)
        folium.GeoJson(force_geom, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 4}).add_to(m2)
        folium.GeoJson(clipped_lsoas, tooltip=folium.GeoJsonTooltip(fields=['LSOA21NM']), style_function=lambda x: {'fillColor': 'transparent', 'color': '#3186cc', 'weight': 1}).add_to(m2)
        
        map_data_lsoa = st_folium(m2, height=500, width='stretch', key="zoomed_map", returned_objects=["last_active_drawing"])

        if map_data_lsoa and map_data_lsoa.get('last_active_drawing'):
            properties = map_data_lsoa['last_active_drawing'].get('properties', {})
            new_clicked_lsoa = properties.get('LSOA21NM')
            if new_clicked_lsoa and st.session_state.clicked_lsoa != new_clicked_lsoa:
                st.session_state.clicked_lsoa = new_clicked_lsoa
                st.rerun(scope='fragment')

    with col2: 
        milp_ui_sidebar(force_milp_data)

@st.fragment
def dynamic_fragment_container():
    if st.session_state.clicked_force is None: main_layout()
    else: zoomed_lsoa()

def tab1():
    head_and_filt()
    st.markdown("---")
    dynamic_fragment_container()

def explorer_page():
    dataset_explorer.crime_dataset_explorer(df)

def graphs_page():
    graphs.tab3(df)

pg = st.navigation([
    st.Page(tab1, title="Map", default=True),
    st.Page(explorer_page, title="Explorer", url_path="explorer"),
    st.Page(graphs_page, title="Graphs", url_path="graphs")
], position="top")

pg.run()