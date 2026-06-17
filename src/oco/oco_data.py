# Data preparation
import pandas as pd
import numpy as np
import calendar
from datetime import date as date_func

print("DATA PREPARATION")

# Load raw data
df_raw = pd.read_parquet("master_final_final.parquet")
print(f"Raw rows: {len(df_raw):,}")

# Filter to your six selected crime types
selected_crimes = [
    "Anti-social behaviour",
    "Public order",
    "Robbery",
    "Shoplifting",
    "Theft from the person",
    "Violence and sexual offences"
]

df = df_raw[df_raw['Crime type'].isin(selected_crimes)].copy()
print(f"Filtered rows (selected crimes): {len(df):,}")

# Create free_time feature (from calendar)
def get_free_time(month_str):
    d = pd.to_datetime(month_str)
    y, m = d.year, d.month
    _, ndays = calendar.monthrange(y, m)
    weekend = 0
    for day in range(1, ndays+1):
        if date_func(y, m, day).weekday() >= 5:
            weekend += 1
    if m == 1: hol = 1
    elif m == 4: hol = 1
    elif m == 5: hol = 1
    elif m == 8: hol = 1
    elif m == 12: hol = 2
    else: hol = 0
    return weekend + hol

uniq_months = df['Month'].unique()
month_free = {m: get_free_time(m) for m in uniq_months}
df['free_time'] = df['Month'].map(month_free)
print("free_time created")

# Create month string column
df['Month_str'] = pd.to_datetime(df['Month']).dt.strftime('%Y-%m')

# Aggregate crime counts per LSOA per month 
monthly_crime = df.groupby(['LSOA code', 'Month_str'], observed=True)['Crime_Count'].sum().reset_index()
monthly_crime.columns = ['LSOA code', 'Month', 'total_crimes']
print(f"Aggregated crime (selected types): {len(monthly_crime):,} rows")

# Get all LSOAs from the RAW data (to keep even zero-crime LSOAs)
all_lsoas = df_raw['LSOA code'].unique()
all_months = sorted(monthly_crime['Month'].unique())
print(f"All LSOAs from raw data: {len(all_lsoas):,}")
print(f"Month range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

# Build full panel (all LSOAs × all months)
panel = pd.MultiIndex.from_product([all_lsoas, all_months],
                                   names=['LSOA code', 'Month']).to_frame(index=False)
print(f"Full panel rows: {len(panel):,}")

# Merge crime counts (left join, fill missing with 0)
monthly = panel.merge(monthly_crime, on=['LSOA code', 'Month'], how='left')
monthly['total_crimes'] = monthly['total_crimes'].fillna(0).astype(int)
print(f"After merging crime: {len(monthly):,} rows")

# Extract month-level features 
month_features = df_raw[['Month_str', 'Mean_Temperature_C', 'free_time']].drop_duplicates(subset=['Month_str']).copy()
month_features.columns = ['Month', 'Mean_Temperature_C', 'free_time']
print(f"Month features shape: {month_features.shape} (should be {len(all_months)})")

# Extract LSOA to Police Force mapping (from raw data)
police_map = df_raw[['LSOA code', 'Police Territory']].drop_duplicates(subset=['LSOA code']).copy()
police_map.columns = ['LSOA code', 'Police_Force']
print(f"Police map shape: {police_map.shape} (should be {len(all_lsoas)})")

# Merge features into panel
monthly = monthly.merge(month_features, on='Month', how='left')
monthly = monthly.merge(police_map, on='LSOA code', how='left')
print(f"After merging features: {len(monthly):,} rows")

# Drop rows with missing essential data
before_drop = len(monthly)
monthly = monthly.dropna(subset=['Police_Force', 'Mean_Temperature_C', 'free_time'])
print(f"Dropped {before_drop - len(monthly):,} rows with missing values")
print(f"Unique LSOAs after cleaning: {monthly['LSOA code'].nunique():,}")

# Sort and create lag features
monthly = monthly.sort_values(['LSOA code', 'Month'])
monthly['lag1'] = monthly.groupby('LSOA code')['total_crimes'].shift(1)
monthly['lag12'] = monthly.groupby('LSOA code')['total_crimes'].shift(12)

before_lag = len(monthly)
monthly = monthly.dropna(subset=['lag1', 'lag12'])
print(f"Dropped {before_lag - len(monthly):,} rows with missing lags")
print(f"Final rows: {len(monthly):,}")

# Save prepared data
monthly.to_parquet("prepared_data.parquet", index=False)
monthly.to_csv("prepared_data.csv", index=False)

print("\nData preparation complete.")
print(f"Date range: {monthly['Month'].min()} to {monthly['Month'].max()}")
print(f"Unique LSOAs: {monthly['LSOA code'].nunique():,}")
print(f"Temperature range: {monthly['Mean_Temperature_C'].min():.1f}°C to {monthly['Mean_Temperature_C'].max():.1f}°C")
