import polars as pl
import pandas as pd

def process_and_merge_population():

    master_df = pl.read_csv("midterm_prototype_dataset.csv")

    excel_file = "policeforceareas1991to2024.xlsx"

    # Using pandas to go inside the excel file
    pd_part1 = pd.read_excel(excel_file, sheet_name="Mid-2011 to Mid-2020", skiprows=3)
    pd_part2 = pd.read_excel(excel_file, sheet_name="Mid-2021 to Mid-2024", skiprows=3)
    
    ons_part1 = pl.from_pandas(pd_part1)
    ons_part2 = pl.from_pandas(pd_part2)
    
    # Combine two tables into one
    ons_combined = pl.concat([ons_part1, ons_part2])
    
    # Take all the relevant columns (skipping Code, Name, and Year)
    age_sex_columns = ons_combined.columns[3:]
    
    # Sum all the entries from a row into one entry (one column)
    pop_df = ons_combined.with_columns(pl.sum_horizontal(age_sex_columns).alias("Population"))
    
    # Keep relevant columns
    pop_df = pop_df.select(["PFA 2023 Name", "Year", "Population"])
    
    # Match the names to the names in master dataset
    pop_df = pop_df.with_columns(
        pl.col("PFA 2023 Name")
        .str.to_lowercase()
        .str.replace(" police", "")
        .str.replace(" & ", " and ")
        .str.replace(" ", "-")
        .str.replace(" ", "-")
        .alias("Police_Force")
    )
    
    # Extract 2018 from 2018-07 so the datasets have a matching column
    # str.slice(0,4) takes the first 4 numbers so 2018
    master_df = master_df.with_columns(pl.col("Month").str.slice(0, 4).cast(pl.Int64).alias("Year"))
    
    # Join them based on the Police Force Area and the Year
    final_df = master_df.join(pop_df, on=["Police_Force", "Year"], how="left")
    
    # (Total Crimes / Population) * 1000
    final_df = final_df.with_columns(((pl.col("Total_Crimes") / pl.col("Population")) * 1000).alias("Crime_Rate_Per_1000"))
    
    output_name = "final_midterm_prototype_with_rates.csv"
    final_df.write_csv(output_name)
    print(f"\nMaster dataset saved as: {output_name}")

if __name__ == "__main__":
    process_and_merge_population()