import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_side_by_side_map():
    repo_root = Path(__file__).resolve().parents[2]
    uk_map = gpd.read_file(repo_root / "data" / "SHP" / "Police_Force_Areas_UK.shp")
    
    uk_map['Police_Force'] = (
        uk_map['PFANM']
        .str.lower()
        .str.replace(" police", "")
        .str.replace(" & ", " and ")
        .str.replace(" ", "-")
        .str.replace(" ", "-")      
    )
    
    df = pd.read_csv(repo_root / "data" / "final_midterm_prototype_with_rates.csv")
    
    target_month = "2018-07"
    month_data = df[df['Month'] == target_month]

    month_data = month_data.dropna(subset=['Crime_Rate_Per_1000'])
   
    map_data = uk_map.merge(month_data, on="Police_Force", how="left")
    
    # Create a figure with 2 subplots 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Prototype: {target_month}", fontsize=20, fontweight='bold')
    
    # Temperature Heatmap (Red/Orange)
    map_data.plot(
        column='Mean_Temp', 
        cmap='OrRd',         
        linewidth=0.5, 
        ax=ax1, 
        edgecolor='0.5', 
        legend=True,
        legend_kwds={'label': "Mean Temp (°C)", 'orientation': "horizontal"},
        missing_kwds={'color': 'lightgrey'}
    )
    ax1.set_title('Regional Temperature', fontsize=15)
    ax1.axis('off') 
    
    # Crime Heatmap (Purple/Dark)
    map_data.plot(
        column='Crime_Rate_Per_1000', 
        cmap='Purples',      
        linewidth=0.5, 
        ax=ax2, 
        edgecolor='0.5', 
        legend=True,
        legend_kwds={'label': "Crime Rate (per 1000 people)", 'orientation': "horizontal"},
        missing_kwds={'color': 'lightgrey'}
    )
    ax2.set_title('Crime Rate per 1000 people', fontsize=15)
    ax2.axis('off')
    
    plt.tight_layout()
    output_dir = repo_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img = output_dir / f"midterm_map_per_capita{target_month}.png"
    plt.savefig(output_img, dpi=300) 
    print(f"\nImage saved as {output_img}")

if __name__ == "__main__":
    plot_side_by_side_map()