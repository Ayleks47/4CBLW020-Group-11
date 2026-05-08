import polars as pl

def build_master_dataset():
    crime_lazy = pl.scan_parquet("merged_crime_dataset.parquet")
    
    # Count total crimes per month per police force
    monthly_crime = (
        crime_lazy
        .group_by(["Month", "Police_Force"])
        .agg(pl.len().alias("Total_Crimes"))
        .collect() 
    )
    
    temp_df = pl.read_csv("regional_temperature_2010_2024.csv")
    sun_df = pl.read_csv("regional_sunshine_2010_2024.csv")
    rain_df = pl.read_csv("regional_rainfall_2010_2024.csv") 
    
    master_df = (
        monthly_crime
        .join(temp_df, on=["Month", "Police_Force"], how="left")
        .join(sun_df, on=["Month", "Police_Force"], how="left")
        .join(rain_df, on=["Month", "Police_Force"], how="left")
        .drop_nulls() 
    )
    
    output_name = "midterm_prototype_dataset.csv"
    master_df.write_csv(output_name)
    print(f"Created master dataset: {output_name}")

if __name__ == "__main__":
    build_master_dataset()