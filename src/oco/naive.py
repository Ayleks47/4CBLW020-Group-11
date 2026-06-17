import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# 1. Load OCO predictions
oco = pd.read_parquet("oco_outlier_optimised.parquet")
print("OCO columns:", oco.columns.tolist())
oco = oco.rename(columns={
    'LSOA code': 'LSOA_code',
    'pred': 'predicted_crime',
    'abs_err': 'abs_error'
})
oco = oco[['LSOA_code', 'Month', 'total_crimes', 'predicted_crime', 'abs_error']]
oco = oco.drop_duplicates(['LSOA_code', 'Month'])

# 2. Load prepared data 
prep = pd.read_parquet("prepared_data.parquet")
print("Prepared columns:", prep.columns.tolist())
# Ensure LSOA code column is consistent
if 'LSOA code' in prep.columns:
    prep = prep.rename(columns={'LSOA code': 'LSOA_code'})
prep = prep[['LSOA_code', 'Month', 'lag12']]
prep = prep.drop_duplicates(['LSOA_code', 'Month'])

# 3. Align month formats
oco['Month'] = pd.to_datetime(oco['Month']).dt.strftime('%Y-%m')
prep['Month'] = pd.to_datetime(prep['Month']).dt.strftime('%Y-%m')

# 4. Merge
merged = oco.merge(prep, on=['LSOA_code', 'Month'], how='inner', validate='one_to_one')
print("Merged rows:", len(merged))

# 5. Compute naive errors 
merged['naive_err'] = np.abs(merged['total_crimes'] - merged['lag12'])
oco_mae = merged['abs_error'].mean()
naive_mae = merged['naive_err'].mean()

print("\n=== OCO vs Naive Seasonal ===")
print(f"Naive MAE: {naive_mae:.2f}")
print(f"OCO MAE:   {oco_mae:.2f}")
print(f"Improvement: {(naive_mae - oco_mae) / naive_mae * 100:.1f}%")

# 6. MAPE 
mask = merged['total_crimes'] > 0
if mask.any():
    oco_mape = (merged.loc[mask, 'abs_error'] / merged.loc[mask, 'total_crimes']).mean() * 100
    naive_mape = (merged.loc[mask, 'naive_err'] / merged.loc[mask, 'total_crimes']).mean() * 100
else:
    oco_mape = naive_mape = np.nan
print(f"\nNaive MAPE (actual>0): {naive_mape:.1f}%")
print(f"OCO MAPE (actual>0):   {oco_mape:.1f}%")

# 7. Statistical tests
t, p_ttest = ttest_rel(merged['naive_err'], merged['abs_error'])
w, p_wilcox = wilcoxon(merged['naive_err'], merged['abs_error'])
print("\n--- Statistical tests ---")
print(f"Paired t-test: t={t:.3f}, p={p_ttest:.6f}")
print(f"Wilcoxon: W={w:.0f}, p={p_wilcox:.6f}")
if p_ttest < 0.05:
    print("OCO significantly better (p<0.05)")

# 9. Error by crime level
bins = [0,5,10,20,50,100,500,5000, merged['total_crimes'].max()]
labels = ['1-5','6-10','11-20','21-50','51-100','101-500','501-5000','5000+']
merged['bin'] = pd.cut(merged['total_crimes'], bins=bins, labels=labels)
grouped = merged.groupby('bin', observed=False).agg(
    count=('total_crimes','count'),
    naive_mae=('naive_err','mean'),
    oco_mae=('abs_error','mean')
).round(2)
print("\nMAE by crime level:")
print(grouped)

# 10. Bar chart
grouped[['naive_mae', 'oco_mae']].plot(kind='bar', figsize=(10,5))
plt.xlabel('Monthly crime level')
plt.ylabel('Mean absolute error')
plt.title('Error by crime level: Naive vs OCO')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mae_by_crime_level.png', dpi=150)
plt.show()

print("\nDone.")
