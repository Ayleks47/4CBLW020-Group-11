import geopandas as gpd
import json
import warnings
from pathlib import Path

# Suppress harmless warnings for cleaner output
warnings.filterwarnings("ignore")

print("1. Loading LSOA shapefile...")
repo_root = Path(__file__).resolve().parents[2]
lsoa_gdf = gpd.read_file(repo_root / "data" / "SHP" / "LSOA" / "LSOA.shp")

# Find the exact column name your shapefile uses for LSOA Names
name_col = None
for col in ['LSOA21NM', 'LSOA11NM', 'LSOA_name', 'LSOA name']:
    if col in lsoa_gdf.columns:
        name_col = col
        break

print(f"2. Using '{name_col}' as the identifier.")
print("3. Calculating spatial intersections (This will take 1 to 3 minutes)...")

# Spatial join to find which LSOAs touch each other
neighbors = gpd.sjoin(lsoa_gdf, lsoa_gdf, how="inner", predicate="intersects")

# Remove self-intersections (LSOAs intersecting themselves)
neighbors = neighbors[neighbors[f'{name_col}_left'] != neighbors[f'{name_col}_right']]

print("4. Building JSON dictionary...")
# Group by the LSOA and create a list of its neighbors
adjacency_dict = neighbors.groupby(f'{name_col}_left')[f'{name_col}_right'].apply(list).to_dict()

output_path = repo_root / "data" / "lsoa_neighbors.json"
with open(output_path, "w") as f:
    json.dump(adjacency_dict, f)

print(f"✅ Success! Adjacency matrix saved to {output_path}")