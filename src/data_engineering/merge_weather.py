import xarray as xr

def combine_weather_data(folder_path: str):
    
    file_pattern = f"{folder_path}/*.nc"
    
    # open_mfdataset stitches all the years together by the time coordinate
    master_cube = xr.open_mfdataset(file_pattern, combine='by_coords')
    
    print("\nMaster Weather Cube")
    # Check if time dimension is actually 180 (15 years * 12 months)
    print(master_cube)
    
    # 
    output_name = "master_rainfall_2010_2024.nc"
    master_cube.to_netcdf(output_name)

if __name__ == "__main__":
    combine_weather_data("weather_data/rainfall")