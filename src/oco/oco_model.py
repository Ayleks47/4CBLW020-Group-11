# OCO model - no rural, weighted, boxcox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

# load data
df = pd.read_parquet("prepared_data.parquet")
print("rows:", len(df))

# features
df['Month'] = pd.to_datetime(df['Month'])
df['month_num'] = df['Month'].dt.month
df['year'] = df['Month'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

monthly_avg = df.groupby('month_num')['temperature'].transform('mean')
df['temp_anomaly'] = df['temperature'] - monthly_avg
df['temp_free'] = df['temperature'] * df['free_time']
df['temp_lag1'] = df['temperature'] * df['lag1']

# boxcox
crime_pos = df['total_crimes'] + 0.01
crime_bc, lam = boxcox(crime_pos)
print("lambda:", lam)

df['lag1_bc'] = boxcox(df['lag1'] + 0.01, lmbda=lam)
df['lag12_bc'] = boxcox(df['lag12'] + 0.01, lmbda=lam)
df['target_bc'] = crime_bc

# features list
feats = ['temp_anomaly', 'free_time', 'temp_free', 'temp_lag1',
         'lag1_bc', 'lag12_bc', 'month_sin', 'month_cos']
df['police_code'] = df['Police_Force'].astype('category').cat.codes
feats.append('police_code')

df_clean = df.dropna(subset=feats + ['target_bc'])
print("clean rows:", len(df_clean))

# scale
scaler = MinMaxScaler()
X = scaler.fit_transform(df_clean[feats])
y = df_clean['target_bc'].values

# sample weights (high crime gets higher weight)
crime_vals = df_clean['total_crimes'].values
weights = 1 + (crime_vals / crime_vals.mean()) ** 0.5
weights = np.clip(weights, 1, 15)

# LSOA mapping
lsoa_codes, lsoa_ids = pd.factorize(df_clean['LSOA_code'])
n_lsoas = len(lsoa_ids)
lsoa_idx_map = {code:i for i,code in enumerate(lsoa_ids)}
lsoa_idx = np.array([lsoa_idx_map[c] for c in df_clean['LSOA_code']])

# OCO class
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
        grad = -2 * we[:,None] * X
        self.W[idx] -= self.lr * grad
        self.W[idx] = np.clip(self.W[idx], -10, 10)
        return pred, err
    def predict_batch(self, idx, X):
        return np.sum(self.W[idx] * X, axis=1)

model = OCO(n_lsoas, len(feats), lr=0.01)
print("parameters:", n_lsoas * len(feats))

# train
batch_size = 50000
n_epochs = 20
print("training...")
for ep in range(n_epochs):
    perm = np.random.permutation(len(X))
    Xp = X[perm]
    yp = y[perm]
    idxp = lsoa_idx[perm]
    wp = weights[perm]
    loss = 0
    nproc = 0
    for start in range(0, len(X), batch_size):
        end = min(start+batch_size, len(X))
        _, err = model.update_batch(idxp[start:end], Xp[start:end], yp[start:end], wp[start:end])
        batch_loss = np.mean(err**2)
        loss = (loss * nproc + batch_loss * (end-start)) / (nproc + (end-start))
        nproc = end
    if (ep+1) % 5 == 0 or ep == 0:
        samp = min(50000, len(X))
        pred_bc = model.predict_batch(lsoa_idx[:samp], X[:samp])
        pred_orig = (pred_bc * lam + 1) ** (1/lam) - 0.01
        act_orig = (y[:samp] * lam + 1) ** (1/lam) - 0.01
        mae = np.mean(np.abs(pred_orig - act_orig))
        print(f"epoch {ep+1}: loss={loss:.4f}, sample MAE={mae:.2f}")

# predict all
pred_bc = model.predict_batch(lsoa_idx, X)
pred_orig = (pred_bc * lam + 1) ** (1/lam) - 0.01
df_clean['pred'] = pred_orig
df_clean['err'] = df_clean['total_crimes'] - df_clean['pred']
df_clean['abs_err'] = np.abs(df_clean['err'])

# eval
mae = df_clean['abs_err'].mean()
rmse = np.sqrt((df_clean['err']**2).mean())
mape = (df_clean['abs_err'] / (df_clean['total_crimes']+0.01)).mean() * 100
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

# baseline (predict mean)
base_mae = np.mean(np.abs(df_clean['total_crimes'] - df_clean['total_crimes'].mean()))
improve = (base_mae - mae) / base_mae * 100
print(f"baseline MAE: {base_mae:.2f}, improvement: {improve:.1f}%")

# temperature sensitivity
base_crime = np.median(df_clean['total_crimes'])
sens = {}
for i, lsoa in enumerate(lsoa_ids):
    raw = model.W[i, 0]
    val = raw * base_crime / 10
    sens[lsoa] = np.clip(val, -1, 1)
df_clean['temp_sens'] = df_clean['LSOA_code'].map(sens)

# LSOA summary
summary = df_clean.groupby(['LSOA_code','Police_Force']).agg({
    'temp_sens':'first',
    'total_crimes':'mean',
    'pred':'mean',
    'abs_err':'mean'
}).rename(columns={'total_crimes':'avg_crime','pred':'avg_pred','abs_err':'avg_err'}).reset_index()
summary = summary.sort_values('temp_sens', ascending=False)
summary['rank'] = range(1, len(summary)+1)

print("\nTop 20 temp-sensitive LSOAs:")
print(summary[['rank','LSOA_code','Police_Force','avg_crime','temp_sens']].head(20))

# save
save_cols = ['LSOA_code','Police_Force','Month','year','month_num',
             'temperature','free_time','temp_sens','total_crimes','pred','abs_err']
df_clean[save_cols].to_parquet("oco_outlier_optimised.parquet", index=False)
summary.to_csv("oco_outlier_lsoa_rankings.csv", index=False)

# simple plot
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sample = df_clean.sample(min(10000, len(df_clean)))
ax1.scatter(sample['total_crimes'], sample['pred'], alpha=0.3, s=5)
ax1.plot([0,sample['total_crimes'].max()], [0,sample['total_crimes'].max()], 'r--')
ax1.set_xlabel('actual'); ax1.set_ylabel('predicted')
ax1.set_title(f'MAE={mae:.2f}')

bins = [0,10,50,100,200,500,1000,5000]
df_clean['bin'] = pd.cut(df_clean['total_crimes'], bins=bins)
bin_err = df_clean.groupby('bin')['abs_err'].mean()
ax2.bar(range(len(bin_err)), bin_err.values)
ax2.set_xticks(range(len(bin_err)))
ax2.set_xticklabels([f"{int(b.left)}-{int(b.right)}" for b in bin_err.index], rotation=45)
ax2.set_xlabel('crime level'); ax2.set_ylabel('mean abs error')
plt.tight_layout()
plt.savefig('oco_outlier_results.png', dpi=150)
plt.show()

print("done")