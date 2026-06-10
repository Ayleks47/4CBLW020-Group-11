# compare OCO vs naive seasonal (same month last year)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# load OCO predictions
oco = pd.read_parquet("src\oco\oco_outlier_optimised.parquet")
print("OCO columns:", oco.columns.tolist())
print("rows:", len(oco))
print("OCO MAE:", oco['abs_error'].mean())

oco = oco[['LSOA_code', 'Month', 'total_crimes', 'predicted_crime', 'abs_error']]
oco = oco.drop_duplicates(['LSOA_code', 'Month'])

# load prepared data (has lag12)
prep = pd.read_parquet("src\oco\prepared_data.parquet")
prep = prep[['LSOA_code', 'Month', 'lag12']]
prep = prep.drop_duplicates(['LSOA_code', 'Month'])

# align months
oco['Month'] = pd.to_datetime(oco['Month']).dt.strftime('%Y-%m')
prep['Month'] = pd.to_datetime(prep['Month']).dt.strftime('%Y-%m')

# merge
merged = oco.merge(prep, on=['LSOA_code', 'Month'], how='inner', validate='one_to_one')
print("merged rows:", len(merged))

# naive errors
merged['naive_err'] = abs(merged['total_crimes'] - merged['lag12'])
oco_mae = merged['abs_error'].mean()
naive_mae = merged['naive_err'].mean()

print("\n=== OCO vs Naive Seasonal ===")
print(f"Naive MAE: {naive_mae:.2f}")
print(f"OCO MAE:   {oco_mae:.2f}")
print(f"Improvement: {(naive_mae - oco_mae)/naive_mae*100:.1f}%")

# MAPE
oco_mape = (merged['abs_error'] / (merged['total_crimes']+0.01)).mean()*100
naive_mape = (merged['naive_err'] / (merged['total_crimes']+0.01)).mean()*100
print(f"\nNaive MAPE: {naive_mape:.1f}%")
print(f"OCO MAPE:   {oco_mape:.1f}%")

# t-test and wilcoxon
t, p_ttest = ttest_rel(merged['naive_err'], merged['abs_error'])
w, p_wilcox = wilcoxon(merged['naive_err'], merged['abs_error'])
print("\n--- Statistical tests ---")
print(f"Paired t-test: t={t:.3f}, p={p_ttest:.6f}")
print(f"Wilcoxon: W={w:.0f}, p={p_wilcox:.6f}")
if p_ttest < 0.05:
    print("✅ OCO significantly better (p<0.05)")

# boxplot
plt.figure(figsize=(8,6))
plt.boxplot([merged['naive_err'], merged['abs_error']], tick_labels=['Naive', 'OCO'], patch_artist=True)
plt.yscale('log')
plt.ylabel('Absolute error (crimes)')
plt.title(f'OCO improves MAE by {((naive_mae-oco_mae)/naive_mae*100):.1f}%')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_compare_box.png', dpi=150)
plt.show()

# error by crime level
bins = [0,5,10,20,50,100,500,5000,merged['total_crimes'].max()]
labels = ['1-5','6-10','11-20','21-50','51-100','101-500','501-5000','5000+']
merged['bin'] = pd.cut(merged['total_crimes'], bins=bins, labels=labels)
grouped = merged.groupby('bin', observed=False).agg(
    count=('total_crimes','count'),
    naive_mae=('naive_err','mean'),
    oco_mae=('abs_error','mean')
).round(2)
print("\nMAE by crime level:")
print(grouped)

# bar chart
grouped[['naive_mae','oco_mae']].plot(kind='bar', figsize=(10,5))
plt.xlabel('Monthly crime level')
plt.ylabel('Mean absolute error')
plt.title('Error by crime level: Naive vs OCO')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mae_by_crime_level.png', dpi=150)
plt.show()

print("\nDone.")