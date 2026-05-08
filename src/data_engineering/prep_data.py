import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile

def stream_zip_to_parquet(zip_file_name: str, target_file_inside_zip: str, output_parquet_name: str):
    print(f"Opening {zip_file_name}...")
    
    with zipfile.ZipFile(zip_file_name, 'r') as z:
        
        # Verify the file exists exactly as named
        if target_file_inside_zip not in z.namelist():
            print(f"\nError: Could not find '{target_file_inside_zip}' in the ZIP.")
            for item in z.namelist():
                print(f" - {item}")
            return
        
        with z.open(target_file_inside_zip) as f:
            csv_iterator = pd.read_csv(
                f,
                chunksize=250_000, 
                usecols=["source_file", "Crime ID", "LSOA code", "Crime type"],
                dtype=str,
                encoding_errors="replace"
            )
            
            parquet_writer = None
            chunk_count = 1
            
            for chunk in csv_iterator:
                print(f"Processing chunk {chunk_count} (approx {chunk_count * 250_000} rows)...")
                
                # Clean missing data
                chunk = chunk.dropna(subset=["source_file", "Crime type"])
                
                # Extract Month and Police Force
                chunk["Month"] = chunk["source_file"].str.extract(r"(\d{4}-\d{2})")

                chunk["Police_Force"] = chunk["source_file"].str.extract(r"\d{4}-\d{2}-(.*?)-street\.csv")
                
                # Drop the source_file column to save space
                chunk = chunk.drop(columns=["source_file"])
                
                # Write to Parquet
                table = pa.Table.from_pandas(chunk)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_parquet_name, table.schema, compression='snappy')
                parquet_writer.write_table(table)
                
                chunk_count += 1
                
            if parquet_writer:
                parquet_writer.close()

if __name__ == "__main__":

    zip_name = "merged_data.zip" 
    
    target_file = "merged_data/merged_street.csv" 
    
    stream_zip_to_parquet(
        zip_file_name=zip_name, 
        target_file_inside_zip=target_file,
        output_parquet_name="merged_crime_dataset.parquet"
    )