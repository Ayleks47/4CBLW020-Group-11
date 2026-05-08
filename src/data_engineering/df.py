import os
import pandas as pd

# Paths of alldata and merged_data
source_folder   = "alldata"
dest_folder = "merged_data"

# Create merged_data folder
os.makedirs(dest_folder, exist_ok=True)

# To check if column names are added to the csv file but only once for per category
is_col_name_added = {"outcomes": False, "stop_and_search": False, "street": False}

# Check the alldata folder and its subfolders and to categories
for folder_path, subfolders, files in os.walk(source_folder):
    for source_file_name in sorted(files):

        # Catagorize the csv files according to file types 
        if source_file_name.endswith("outcomes.csv"):
            file_type = "outcomes"
        elif source_file_name.endswith("stop-and-search.csv"):
            file_type = "stop_and_search"
        elif source_file_name.endswith("street.csv"):
            file_type = "street"
        else:
            continue  

        # Join csv file paths with file names from the source
        source_path = os.path.join(folder_path, source_file_name)

        # Join dest_folder with merged csv files according to the category 
        dest_path = os.path.join(dest_folder, f"merged_{file_type}.csv")


        # Merge rows with chunksize 500000 to avoid any memory error
        # dtype=str to avoid type error
        for subsection_rows in pd.read_csv(source_path, dtype=str, chunksize=50000,
                                on_bad_lines="warn", encoding_errors="replace"):

            # For each subsection add the original files name into a column to new merged file 
            subsection_rows.insert(0, "source_file", source_file_name)

            # Create csv files and check whether or not the first line contains column names
            subsection_rows.to_csv(dest_path, mode="a", index=False,
                        header=not is_col_name_added[file_type])

            is_col_name_added[file_type] = True

