
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')   

print("OCO CRIME PREDICTION MODEL")

# Load prepared data
df = pd.read_parquet("C:/Users/maria/Desktop/OCO/prepared_data.parquet")
print(f"Loaded rows: {len(df):,}")
print(f"Unique LSOAs: {df['LSOA code'].nunique():,}")
print(f"Date range: {df['Month'].min()} to {df['Month'].max()}")

# Feature engineering
df['Month'] = pd.to_datetime(df['Month'])
df['month_num'] = df['Month'].dt.month
df['year'] = df['Month'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

monthly_avg = df.groupby('month_num')['Mean_Temperature_C'].transform('mean')
df['temp_anomaly'] = df['Mean_Temperature_C'] - monthly_avg
df['temp_free'] = df['Mean_Temperature_C'] * df['free_time']
df['temp_lag1'] = df['Mean_Temperature_C'] * df['lag1']

# Box‑Cox transform
crime_pos = df['total_crimes'] + 0.01
crime_bc, lam = boxcox(crime_pos)
print(f"Box-Cox lambda: {lam:.4f}")

df['lag1_bc'] = boxcox(df['lag1'] + 0.01, lmbda=lam)
df['lag12_bc'] = boxcox(df['lag12'] + 0.01, lmbda=lam)
df['target_bc'] = crime_bc

# Feature list and cleaning
feats = ['temp_anomaly', 'free_time', 'temp_free', 'temp_lag1',
         'lag1_bc', 'lag12_bc', 'month_sin', 'month_cos']
df['police_code'] = df['Police_Force'].astype('category').cat.codes
feats.append('police_code')

df_clean = df.dropna(subset=feats + ['target_bc'])
print(f"Clean rows: {len(df_clean):,}")

# Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(df_clean[feats])
y = df_clean['target_bc'].values

# Sample weights
crime_vals = df_clean['total_crimes'].values
weights = 1 + (crime_vals / crime_vals.mean()) ** 0.5
weights = np.clip(weights, 1, 15)

# LSOA mapping
lsoa_codes, lsoa_ids = pd.factorize(df_clean['LSOA code'])
n_lsoas = len(lsoa_ids)
lsoa_idx_map = {code: i for i, code in enumerate(lsoa_ids)}
lsoa_idx = np.array([lsoa_idx_map[c] for c in df_clean['LSOA code']])

#OCO model class
class OCO:
    def __init__(self, n, nf, lr=0.01):
        self.n = n
        self.nf = nf
        self.lr = lr
        self.W = np.zeros((n, nf), dtype=np.float32)
    def update_batch(self, idx, X, y, w):
        pred = np.sum(self.W[idx] * X, axis=1)
        err = y - pred
        we = err * w
        we = np.clip(we, -5, 5)
        grad = -2 * we[:, None] * X
        self.W[idx] -= self.lr * grad
        self.W[idx] = np.clip(self.W[idx], -10, 10)
        return pred, err
    def predict_batch(self, idx, X):
        return np.sum(self.W[idx] * X, axis=1)

model = OCO(n_lsoas, len(feats), lr=0.01)
print(f"Parameters: {n_lsoas * len(feats):,}")

# Training
batch_size = 50000   
n_epochs = 20
print("Training...")
for ep in range(n_epochs):
    perm = np.random.permutation(len(X))
    Xp = X[perm]
    yp = y[perm]
    idxp = lsoa_idx[perm]
    wp = weights[perm]
    loss = 0
    nproc = 0
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        _, err = model.update_batch(idxp[start:end], Xp[start:end], yp[start:end], wp[start:end])
        batch_loss = np.mean(err ** 2)
        loss = (loss * nproc + batch_loss * (end - start)) / (nproc + (end - start))
        nproc = end
    if (ep + 1) % 5 == 0 or ep == 0:
        samp = min(50000, len(X))
        pred_bc = model.predict_batch(lsoa_idx[:samp], X[:samp])
        pred_orig = (pred_bc * lam + 1) ** (1 / lam) - 0.01
        act_orig = (y[:samp] * lam + 1) ** (1 / lam) - 0.01
        mae = np.mean(np.abs(pred_orig - act_orig))
        print(f"Epoch {ep+1}: loss={loss:.4f}, sample MAE={mae:.2f}")

# Historical predictions and evaluation
pred_bc = model.predict_batch(lsoa_idx, X)
pred_orig = (pred_bc * lam + 1) ** (1 / lam) - 0.01
df_clean['pred'] = pred_orig
df_clean['err'] = df_clean['total_crimes'] - df_clean['pred']
df_clean['abs_err'] = np.abs(df_clean['err'])

# Standard metrics
mae = df_clean['abs_err'].mean()
rmse = np.sqrt((df_clean['err'] ** 2).mean())

# MAPE
mask_nonzero = df_clean['total_crimes'] > 0
if mask_nonzero.any():
    mape_nonzero = (df_clean.loc[mask_nonzero, 'abs_err'] / df_clean.loc[mask_nonzero, 'total_crimes']).mean() * 100
else:
    mape_nonzero = np.nan

# Symmetric Mean Absolute Percentage Error
denom = (np.abs(df_clean['total_crimes']) + np.abs(df_clean['pred'])) / 2
smape = np.mean(np.abs(df_clean['err']) / denom) * 100

print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
print(f"MAPE (actual > 0 only): {mape_nonzero:.1f}%")
print(f"sMAPE: {smape:.1f}%")

baseline_mae = np.mean(np.abs(df_clean['total_crimes'] - df_clean['total_crimes'].mean()))
improve = (baseline_mae - mae) / baseline_mae * 100
print(f"Baseline MAE: {baseline_mae:.2f}, improvement: {improve:.1f}%")

#Temperature sensitivity
base_crime = np.median(df_clean['total_crimes'])
sens = {}
for i, lsoa in enumerate(lsoa_ids):
    raw = model.W[i, 0]
    val = raw * base_crime / 10
    sens[lsoa] = np.clip(val, -1, 1)
df_clean['temp_sens'] = df_clean['LSOA code'].map(sens)

# LSOA summary
summary = df_clean.groupby(['LSOA code', 'Police_Force']).agg({
    'temp_sens': 'first',
    'total_crimes': 'mean',
    'pred': 'mean',
    'abs_err': 'mean'
}).rename(columns={'total_crimes': 'avg_crime', 'pred': 'avg_pred', 'abs_err': 'avg_err'}).reset_index()
summary = summary.sort_values('temp_sens', ascending=False)
summary['rank'] = range(1, len(summary) + 1)

print("\nTop 20 temperature-sensitive LSOAs:")
print(summary[['rank', 'LSOA code', 'Police_Force', 'avg_crime', 'temp_sens']].head(20))

# Save historical results
save_cols = ['LSOA code', 'Police_Force', 'Month', 'year', 'month_num',
             'Mean_Temperature_C', 'free_time', 'temp_sens', 'total_crimes', 'pred', 'abs_err']
df_clean[save_cols].to_parquet("oco_outlier_optimised.parquet", index=False)
summary.to_csv("oco_outlier_lsoa_rankings.csv", index=False)

# Forecast next month 
print("\nGenerating next month forecast...")

# Precompute police code mapping 
police_code_map = df_clean.set_index('LSOA code')['police_code'].to_dict()

# Determine target month
last_month_date = df_clean['Month'].max()
next_month_date = last_month_date + pd.DateOffset(months=1)
target_month_num = next_month_date.month

# Precompute historical average temperature for the target month
hist_avg_temp = df_clean[df_clean['month_num'] == target_month_num]['Mean_Temperature_C'].mean()

# Get last month's crime 
last_month_data = df_clean[df_clean['Month'] == last_month_date]
last_crime = dict(zip(last_month_data['LSOA code'], last_month_data['total_crimes']))

# Get same month last year 
last_year_same_month = last_month_date - pd.DateOffset(years=1)
lag12_data = df_clean[df_clean['Month'] == last_year_same_month]
last_year_crime = dict(zip(lag12_data['LSOA code'], lag12_data['total_crimes']))

# Forecast inputs 
forecast_temp = 20.0        
forecast_free = 10          

# Precompute cyclic month features
month_sin = np.sin(2 * np.pi * target_month_num / 12)
month_cos = np.cos(2 * np.pi * target_month_num / 12)

next_month_features = []
lsoa_list = []
total = len(lsoa_idx_map)
print(f"Processing {total} LSOAs...")

for i, (lsoa_code, idx) in enumerate(lsoa_idx_map.items()):
    if i % 5000 == 0:
        print(f"  Progress: {i}/{total} LSOAs")

    lag1 = last_crime.get(lsoa_code, 0)
    lag12 = last_year_crime.get(lsoa_code, 0)

    temp_anomaly = forecast_temp - hist_avg_temp
    temp_free = forecast_temp * forecast_free
    temp_lag1 = forecast_temp * lag1

    lag1_bc = ((lag1 + 0.01) ** lam - 1) / lam if lam != 0 else np.log(lag1 + 0.01)
    lag12_bc = ((lag12 + 0.01) ** lam - 1) / lam if lam != 0 else np.log(lag12 + 0.01)

    # Lookup police code from precomputed dictionary
    police_code = police_code_map[lsoa_code]

    raw_features = np.array([
        temp_anomaly, forecast_free, temp_free, temp_lag1,
        lag1_bc, lag12_bc, month_sin, month_cos, police_code
    ])
    scaled_features = scaler.transform(raw_features.reshape(1, -1))[0]

    next_month_features.append(scaled_features)
    lsoa_list.append(lsoa_code)

print(f"  Progress: {total}/{total} LSOAs – complete.")

X_next = np.array(next_month_features)
pred_next_bc = model.predict_batch(np.arange(len(lsoa_list)), X_next)
pred_next_orig = (pred_next_bc * lam + 1) ** (1 / lam) - 0.01
pred_next_orig = np.maximum(pred_next_orig, 0)

next_month_df = pd.DataFrame({
    'LSOA code': lsoa_list,
    'predicted_crime': pred_next_orig
})

next_month_df.to_csv("next_month_forecast.csv", index=False)
print(f"\nNext month forecast saved to 'next_month_forecast.csv'")
print(f"Forecast month: {next_month_date.strftime('%Y-%m')}")
print(f"Number of LSOAs: {len(next_month_df)}")
    
print("Done.")
