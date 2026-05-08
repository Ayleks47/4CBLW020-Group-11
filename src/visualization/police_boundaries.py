import geopandas as gpd
import matplotlib.pyplot as plt

shapefile_path = "SHP/Police_Force_Areas_UK.shp" 

police_map = gpd.read_file(shapefile_path)

print(police_map.columns)

# Plot the map
police_map.plot(figsize=(10, 10), edgecolor="black", facecolor="lightblue")

plt.title("UK Police Force Areas")
plt.show()
