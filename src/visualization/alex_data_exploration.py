from pathlib import Path 
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"

police_path = data_dir / "SHP" / "Police_Force_Areas_UK.shp"
lsoa_path = data_dir / "SHP" / "LSOA" / "LSOA.shp"
parquet_path = data_dir / "master_dataset_full_set_no_solved_percent.parquet"

sarima = data_dir / "SARIMA_forecast.csv"
oco = data_dir / "oco_outlier_optimised.parquet"
oco_new = data_dir / "oco_forecast.csv"

sarima_df = pd.read_csv(sarima)
oco_df = pd.read_parquet(oco)
oco_df_new = pd.read_csv(oco_new)

sarima_df.rename(columns={sarima_df.columns[0]: 'Index', sarima_df.columns[2]: 'Predicted'}, inplace=True)

print(oco_df.columns)
print(oco_df_new.columns)