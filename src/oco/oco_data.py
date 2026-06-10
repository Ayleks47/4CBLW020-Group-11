# DATA PREP - for new dataset

import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from datetime import date as date_func

print("DATA PREPARATION")

# load
df = pd.read_parquet("src\oco\master_final_4.parquet")
print("rows:", len(df))

# sample for quick checks
df.sample(100000, random_state=42).to_parquet("debug_sample.parquet")

# free time
def get_free_time(month_str):
    d = pd.to_datetime(month_str)
    y, m = d.year, d.month
    _, ndays = calendar.monthrange(y, m)
    weekend = 0
    for day in range(1, ndays+1):
        if date_func(y, m, day).weekday() >= 5:
            weekend += 1
    # simple UK holidays
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

# aggregate to LSOA-month
monthly = df.groupby(['LSOA code', 'Month'], observed=True)['Crime_Count'].sum().reset_index()
monthly.columns = ['LSOA_code', 'Month', 'total_crimes']

# add features (no rural_urban)
feat = df[['LSOA code', 'Month', 'Mean_Temp', 'free_time', 'Police Territory']].drop_duplicates()
feat.columns = ['LSOA_code', 'Month', 'temperature', 'free_time', 'Police_Force']
monthly = monthly.merge(feat, on=['LSOA_code', 'Month'], how='left')
monthly = monthly.dropna(subset=['temperature', 'free_time'])

# lags
monthly = monthly.sort_values(['LSOA_code', 'Month'])
monthly['lag1'] = monthly.groupby('LSOA_code', observed=True)['total_crimes'].shift(1)
monthly['lag12'] = monthly.groupby('LSOA_code', observed=True)['total_crimes'].shift(12)
monthly = monthly.dropna(subset=['lag1', 'lag12'])

# save
monthly.to_parquet("prepared_data.parquet", index=False)
monthly.to_csv("prepared_data.csv", index=False)

print("done. rows:", len(monthly))
print("date range:", monthly['Month'].min(), "to", monthly['Month'].max())
print("unique LSOAs:", monthly['LSOA_code'].nunique())
print("temp range:", monthly['temperature'].min(), "to", monthly['temperature'].max())