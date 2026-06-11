import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr

tourist_crimes = ['Other theft','Public order','Robbery','Theft from the person']

def tourism_correlation():
    crime_df = pl.scan_parquet("merged_crime_dataset.parquet")
    crime_sort = (
        crime_df
        .drop_nulls(subset = ["Month"])
        .filter(pl.col("Month").is_between(pl.lit("2012-01"), pl.lit("2023-12")))
        .filter(pl.col("Crime type").is_in(tourist_crimes))
        .group_by("Month")
        .agg(pl.len().alias("Total_Crimes_National"))
        .sort("Month")
    )
    total_crime = crime_sort.collect().to_pandas()
    total_crime["Month"] = pd.to_datetime(total_crime["Month"])

    #Not really necessary to change the tourism df around but whatever

    tourism_df = pd.read_parquet("Tourism_MergedCorrect.parquet")
    
    tourism_df = (
        tourism_df[["Date", "TourismWorldTotal"]]
        .rename(columns = {
            "Date" : "Month",
            "TourismWorldTotal" : "Tourists"
        })
    )
    tourism_df["Month"] = pd.to_datetime(tourism_df["Month"])
    tourism_df.set_index("Month", inplace = True)

    tourism_decomp = seasonal_decompose(
        tourism_df["Tourists"],
        model = "additive",
        period = 12,
        extrapolate_trend = "freq"
    )

    crime_decomp = seasonal_decompose(
        total_crime["Total_Crimes_National"],
        model = "additive",
        period = 12,
        extrapolate_trend = "freq"
    )

    r_value, p_value = pearsonr(crime_decomp.resid, tourism_decomp.resid)
    print(f"Correlation coeffiecient: {r_value}")
    print(f"Correlation p value: {p_value}")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = crime_decomp.resid
    x = tourism_decomp.resid
    
    ax.scatter(x, y)
    m, c = np.polyfit(x, y, 1)
    ax.plot(x, (m * x) + c, color='red', linewidth=2, label='Linear Trend')
    plt.title("Impact of Tourism on related Crimes (Theft, Robbery, Public Order)")
    plt.xlabel("Tourism Anomaly")
    plt.ylabel("Crime Anomaly")
    plt.axhline(color='gray', linestyle='--')
    plt.axvline(color='gray', linestyle='--')
    ax.text(0.05, 0.95, f'Pearson R: {r_value:.3f}\nP-value: {p_value:.3g}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("tourism_crime_correlation.png")

tourism_correlation()
