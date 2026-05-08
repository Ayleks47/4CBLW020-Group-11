import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

crime_list = ['Anti-social behaviour', 'Bicycle theft', 'Burglary', 'Criminal damage and arson', 'Drugs', 'Other crime', 'Other theft', 'Possession of weapons', 'Public disorder and weapons', 'Public order', 'Robbery', 'Shoplifting', 'Theft from the person', 'Vehicle crime', 'Violence and sexual offences', 'Violent crime']

def generate_variance_df():
    df = pl.scan_parquet("data/merged_crime_dataset.parquet")
    
    sorted_df = (
        df
        .drop_nulls(subset=["Month", "Police_Force"])
        .group_by(["Police_Force", "Month"])
        .agg(pl.len().alias("Total_Crimes"))
        .sort(["Police_Force", "Month"])
        .collect()
    ).to_pandas()

    results = []
    for force in sorted_df['Police_Force'].unique():
        force_data = sorted_df[sorted_df['Police_Force'] == force].copy()
        force_data['Month'] = pd.to_datetime(force_data['Month'])
        force_data.set_index('Month', inplace=True)

        decomp = seasonal_decompose(
                force_data['Total_Crimes'] + 1, 
                model='multiplicative', 
                period=12,
                extrapolate_trend='freq'
            )
        multiplier_amplitude = decomp.seasonal.max() - decomp.seasonal.min()
        results.append({
            'Police_Force': force,
            'Volatility': multiplier_amplitude
        })
            
    return pd.DataFrame(results)

def generate_variance_df_by_crime_type(crime_type):
    df = pl.scan_parquet("data/merged_crime_dataset.parquet")
    
    sorted_df = (
        df
        .drop_nulls(subset=["Month", "Police_Force", "Crime type"])
        .filter(pl.col("Crime type") == crime_type)
        .group_by(["Police_Force", "Month"])
        .agg(pl.len().alias("Total_Crimes"))
        .sort(["Police_Force", "Month"])
        .collect()
    ).to_pandas()

    results = []
    for force in sorted_df['Police_Force'].unique():
        force_data = sorted_df[sorted_df['Police_Force'] == force].copy()
        force_data['Month'] = pd.to_datetime(force_data['Month'])
        force_data.set_index('Month', inplace=True)

        #incase theres not enough data for specific crimes; fix later 
        if len(force_data) >= 24:
            decomp = seasonal_decompose(
                force_data['Total_Crimes'] + 1, 
                model='multiplicative', 
                period=12,
                extrapolate_trend='freq'
            )
            
            multiplier_amplitude = decomp.seasonal.max() - decomp.seasonal.min()
            results.append({
                'Police_Force': force,
                'Volatility': multiplier_amplitude
            })
            
    return pd.DataFrame(results)

def plot_variance_map(df):

    # pretty much just Maciej's work here 
    
    uk_map = gpd.read_file("data/SHP/Police_Force_Areas_UK.shp")
    
    uk_map['Police_Force'] = (
        uk_map['PFANM']
        .str.lower()
        .str.replace(" police", "")
        .str.replace(" & ", " and ")
        .str.replace(" ", "-")      
    )

    map_data = uk_map.merge(df, on = "Police_Force", how = "left")

    fig, ax = plt.subplots(1,1, figsize = (10,10))
    fig.suptitle("Seasonal Volatility Map by Police Force")

    map_data.plot(
        column = "Volatility",
        cmap = "YlOrRd",
        linewidth = 0.25,
        edgecolor = '0',
        ax = ax,

        legend = True,
        legend_kwds = {
            "label" : "Volatility (Max-Min Seasonal Multiple Variance)",
            "orientation" : "horizontal",
            'shrink': 0.7,
            'pad': 0.05
        },
        # scotland does not exist (real and true)
        missing_kwds = {
            "color" : "lightgrey",
        }
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("volatility_map.png")