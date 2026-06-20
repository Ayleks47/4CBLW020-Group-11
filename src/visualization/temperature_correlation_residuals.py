import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr

def temperature_correlation():
    repo_root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(repo_root / "data" / "final_midterm_prototype_with_rates.csv")

    national_df = df.groupby('Month').agg({'Total_Crimes': 'sum','Mean_Temp': 'mean' }).reset_index()

    national_df['Month'] = pd.to_datetime(national_df['Month'])
    national_df = national_df.sort_values('Month').set_index('Month')

    # Reindex to ensure continuous monthly frequency 
    idx = pd.date_range(start=national_df.index.min(), end=national_df.index.max(), freq='MS')
    national_df = national_df.reindex(idx)

    # Cut off at covid
    pre_covid = national_df.loc[:'2019-12'].copy().dropna()

    crime_decomp = seasonal_decompose(
        pre_covid['Total_Crimes'], 
        model='additive', 
        period=12, 
    )
    temp_decomp = seasonal_decompose(
        pre_covid['Mean_Temp'], 
        model='additive', 
        period=12, 
    )

    # Calculate r value
    # Put them in a dataframe to perfectly align indices and drop NaNs
    resid_df = pd.DataFrame({'crime_resid': crime_decomp.resid, 'temp_resid': temp_decomp.resid}).dropna()

    r_value, p_value = pearsonr(resid_df['temp_resid'], resid_df['crime_resid'])
    print(f"Correlation coefficient: {r_value:.3f}")
    print(f"Correlation p-value: {p_value:.3g}")

    # Scatterplot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = resid_df['temp_resid']
    y = resid_df['crime_resid']

    ax.scatter(x, y, alpha=0.6, color='orange')
    m, c = np.polyfit(x, y, 1)
    ax.plot(x, (m * x) + c, color='red', linewidth=2)

    plt.title("Impact of Temperature on Total Crime (2010-2019)")
    plt.xlabel("Temperature Anomaly (Unexpected Heat/Cold)")
    plt.ylabel("Crime Anomaly (Unexpected Crime Volume)")
    plt.axhline(color='gray', linestyle='--')
    plt.axvline(color='gray', linestyle='--')
    
    # Add stats box to chart
    ax.text(0.05, 0.95, f'Pearson R: {r_value:.3f}\nP-value: {p_value:.3g}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    output_dir = repo_root / "outputs" / "Presentation Temperature"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "temp_crime_correlation_precovid.png")

# Run the function
temperature_correlation()