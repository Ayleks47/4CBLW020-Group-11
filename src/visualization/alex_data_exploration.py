from pathlib import Path 
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"

police_path = data_dir / "SHP" / "Police_Force_Areas_UK.shp"
lsoa_path = data_dir / "SHP" / "LSOA" / "LSOA.shp"
parquet_path = data_dir / "master_dataset_full_set_no_solved_percent.parquet"

sarima = data_dir / "SARIMA_forecast.csv"
oco = data_dir / "oco_outlier_optimised.parquet"

sarima_df = pd.read_csv(sarima)
oco_df = pd.read_parquet(oco)
df = pd.read_parquet(parquet_path)

sarima_df = sarima_df.rename(columns={sarima_df.columns[0]: 'Index', sarima_df.columns[1]: 'LSOA code', sarima_df.columns[2]: 'Predicted'}, inplace=True)
oco_df = oco_df.rename(columns={oco_df.columns[0]: 'LSOA code'})

# print(oco_df[oco_df["Month"].between('2020-01-01', '2026-03-01')]['Month'])
print(f"OCO: {oco_df.columns}")
print(f"SARIMA: {sarima_df.columns}")