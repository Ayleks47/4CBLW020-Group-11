import pandas as pd
import calendar
from datetime import date as date_func

print("DATA PREPARATION")

# 1. Load only necessary columns from raw data
cols = ['LSOA code', 'Police Territory', 'Crime type', 'Crime_Count', 'Mean_Temperature_C', 'Month']
df_raw = pd.read_parquet("master_final_final.parquet", columns=cols)
print("raw rows:", len(df_raw))

# 2. Filter to exactly the six selected crime types
selected = [
    "Anti-social behaviour",
    "Public order",
    "Robbery",
    "Shoplifting",
    "Theft from the person",
    "Violence and sexual offences"
]
df = df_raw[df_raw['Crime type'].isin(selected)].copy()
print("filtered rows (six types only):", len(df))
del df_raw  # free memory

# 3. Create Month string and free_time
df['Month'] = df['Month'].astype(str)
df['Month_str'] = df['Month'].str[:7]

def get_free_time(month_str):
    d = pd.to_datetime(month_str)
    y, m = d.year, d.month
    _, ndays = calendar.monthrange(y, m)
    weekend = sum(1 for day in range(1, ndays+1) if date_func(y, m, day).weekday() >= 5)
    holidays = {1:1, 4:1, 5:1, 8:1, 12:2}.get(m, 0)
    return weekend + holidays

months = df['Month_str'].unique()
month_free = {m: get_free_time(m) for m in months}
df['free_time'] = df['Month_str'].map(month_free)

# 4. Aggregate crime counts per LSOA per month
monthly_crime = df.groupby(['LSOA code', 'Month_str'], observed=True)['Crime_Count'].sum().reset_index()
monthly_crime.columns = ['LSOA code', 'Month', 'total_crimes']

# 5. Get all LSOAs and all months from the filtered data 
all_lsoas = df['LSOA code'].unique()
all_months = sorted(df['Month_str'].unique())
print("all LSOAs:", len(all_lsoas))
print("all months:", len(all_months), "from", all_months[0], "to", all_months[-1])

# 6. Build full panel
panel = pd.MultiIndex.from_product([all_lsoas, all_months], names=['LSOA code', 'Month']).to_frame(index=False)
print("panel rows:", len(panel))

# 7. Merge crime counts 
monthly = panel.merge(monthly_crime, on=['LSOA code', 'Month'], how='left')
monthly['total_crimes'] = monthly['total_crimes'].fillna(0).astype(int)

# 8. Month features: temperature  and free_time
# Compute average temperature per month from the filtered df
temp_avg = df.groupby('Month_str')['Mean_Temperature_C'].mean().reset_index()
temp_avg.columns = ['Month', 'Mean_Temperature_C']

month_features = pd.DataFrame({'Month': all_months})
month_features['free_time'] = month_features['Month'].map(month_free)
month_features = month_features.merge(temp_avg, on='Month', how='left')

# 9. Police Force mapping
police_map = df[['LSOA code', 'Police Territory']].drop_duplicates(subset=['LSOA code']).copy()
police_map.columns = ['LSOA code', 'Police_Force']

# 10. Merge features
monthly = monthly.merge(month_features, on='Month', how='left')
monthly = monthly.merge(police_map, on='LSOA code', how='left')

# 11. Drop rows with missing essential data
monthly = monthly.dropna(subset=['Police_Force', 'Mean_Temperature_C', 'free_time'])
print("rows after dropping missing:", len(monthly))

# 12. Sort and create lag features
monthly = monthly.sort_values(['LSOA code', 'Month'])
monthly['lag1'] = monthly.groupby('LSOA code')['total_crimes'].shift(1)
monthly['lag12'] = monthly.groupby('LSOA code')['total_crimes'].shift(12)
monthly = monthly.dropna(subset=['lag1', 'lag12'])
print("final rows:", len(monthly))

# 13. Save
monthly.to_parquet("prepared_data.parquet", index=False)
monthly.to_csv("prepared_data.csv", index=False)

print("\nDone.")
print("date range:", monthly['Month'].min(), "to", monthly['Month'].max())
print("unique LSOAs:", monthly['LSOA code'].nunique())
print("temp range:", monthly['Mean_Temperature_C'].min(), "to", monthly['Mean_Temperature_C'].max())
