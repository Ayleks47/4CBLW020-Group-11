import geopandas as gpd
import xarray as xr
import regionmask
import pandas as pd
from pathlib import Path

def extract_regional_weather():
    repo_root = Path(__file__).resolve().parents[2]
    shapefile_path = repo_root / "data" / "SHP" / "Police_Force_Areas_UK.shp"
    police_map = gpd.read_file(shapefile_path)
    
    # Project to British National Grid to match Met Office data
    police_map = police_map.to_crs("EPSG:27700") 
    
    # Clean the police force names to match the Parquet data
    police_map['Police_Force'] = (
        police_map['PFANM']
        .str.lower()
        .str.replace(" police", "")
        .str.replace(" & ", " and ") 
        .str.replace(" ", "-")      
    )
    
    # tells xarray to use dask to prevent out of memory errors
    rainfall_cube = xr.open_dataset("master_rainfall_2010_2024.nc", chunks={'time': -1})
    
    # Using the exact dimension names from the HadUK-Grid NetCDF
    mask = regionmask.mask_geopandas(
        police_map, 
        rainfall_cube.projection_x_coordinate, 
        rainfall_cube.projection_y_coordinate, 
        wrap_lon=False
    )
    
    # Computing the data
    regional_rainfall = rainfall_cube['rainfall'].groupby(mask).mean()
    
    df_temp = regional_rainfall.to_dataframe().reset_index()
    
    region_names = police_map['Police_Force'].to_dict()
    print(df_temp.columns)
    df_temp['Police_Force'] = df_temp['mask'].map(region_names)
    df_temp['Month'] = df_temp['time'].dt.strftime('%Y-%m')
    
    final_table = df_temp[['Month', 'Police_Force', 'rainfall']].rename(columns={'rainfall': 'Total_Rainfall'})
    final_table = final_table.dropna()
    
    output_name = repo_root / "data" / "regional_rainfall_2010_2024.csv"
    final_table.to_csv(output_name, index=False)
    print(f"saved as: {output_name}")

if __name__ == "__main__":
    extract_regional_weather()