import streamlit as st
import pandas as pd

st.set_page_config(page_title="Crime Dataset Explorer", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

st.title("Crime Dataset Explorer")

DATA_PATH = "../../data/master_dataset_dropped_outcome_nulls.parquet"
df = load_data(DATA_PATH)

st.sidebar.header("Filter dataset")

month_options = sorted(df["Month"].dropna().unique())
selected_months = st.sidebar.multiselect("Month", month_options, default=month_options[:5])

territory_options = sorted(df["Police Territory"].dropna().unique())
with st.sidebar:
    with st.expander("Police Territory"):
        select_all_territories = st.checkbox("Select all territories", value=True)
        selected_territories = []
        for territory in territory_options:
            checked = st.checkbox(territory, value=select_all_territories)
            if checked:
                selected_territories.append(territory)
            

crime_type_options = sorted(df["Crime type"].dropna().unique())
# selected_crime_types = st.sidebar.multiselect("Crime type", crime_type_options, default=crime_type_options)
with st.sidebar:
    with st.expander("Crime Type"):
        select_all_crimes = st.checkbox("Select all crime types", value=True)
        selected_crime_types = []
        for crime in crime_type_options:
            checked = st.checkbox(crime, value=select_all_crimes)
            if checked:
                selected_crime_types.append(crime)

rural_urban_options = sorted(df["Rural_Urban"].dropna().unique())
selected_rural_urban = st.sidebar.multiselect("Rural / Urban", rural_urban_options, default=rural_urban_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Numeric filters")

crime_count_min, crime_count_max = int(df["Crime_Count"].min()), int(df["Crime_Count"].max())
crime_count_range = st.sidebar.slider("Crime Count", crime_count_min, crime_count_max, (crime_count_min, min(crime_count_min + 20, crime_count_max)))

solved_min, solved_max = float(df["Solved_%"].min()), float(df["Solved_%"].max())
solved_range = st.sidebar.slider("Solved %", solved_min, solved_max, (solved_min, solved_max), format="%.1f")

temp_min, temp_max = float(df["Mean_Temp"].min()), float(df["Mean_Temp"].max())
temp_range = st.sidebar.slider("Mean Temperature", temp_min, temp_max, (temp_min, temp_max), format="%.1f")

sun_min, sun_max = float(df["Sunshine_Duration"].min()), float(df["Sunshine_Duration"].max())
sun_range = st.sidebar.slider("Sunshine Duration", sun_min, sun_max, (sun_min, sun_max), format="%.1f")

rain_min, rain_max = float(df["Total_Rainfall"].min()), float(df["Total_Rainfall"].max())
rain_range = st.sidebar.slider("Total Rainfall", rain_min, rain_max, (rain_min, rain_max), format="%.1f")

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

st.markdown("### Filtered dataset ")

# selected_columns = st.multiselect(
#     "Columns to display",
#     options=df.columns.tolist(),
#     default=df.columns.tolist(),
# )

st.dataframe(filtered_df, hide_index=True)