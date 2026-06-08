import streamlit as st

def crime_dataset_explorer(df):
    col1, col2 = st.columns([1, 4])
        
    col2.title("Crime Dataset Explorer")

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
        & df["Mean_Temp"].between(temp_range[0], temp_range[1])
        & df["Sunshine_Duration"].between(sun_range[0], sun_range[1])
        & df["Total_Rainfall"].between(rain_range[0], rain_range[1])
    ]

    col2.markdown("### Filtered dataset ")


    col2.dataframe(filtered_df, hide_index=True)
    st.toast("Done loading tab 2")
