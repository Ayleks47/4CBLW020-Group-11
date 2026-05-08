import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def generate_temp_correlation_map():

    df = pd.read_csv("data/final_midterm_prototype_with_rates.csv")
    
    df = df.dropna(subset=['Crime_Rate_Per_1000', 'Mean_Temp'])

    results = []
    
    for force in df['Police_Force'].unique():
        force_data = df[df['Police_Force'] == force]
        
        # Calculate only if there is enough data
        if len(force_data) >= 24:
            # Calculate correlation between temperature and crime rate
            corr_score = force_data['Mean_Temp'].corr(force_data['Crime_Rate_Per_1000'])
            
            results.append({
                'Police_Force': force,
                'Temp_Crime_Correlation': corr_score
            })
            
    corr_df = pd.DataFrame(results)

    uk_map = gpd.read_file("data/SHP/Police_Force_Areas_UK.shp")
    
    # Standard string cleaning to match our master dataset
    uk_map['Police_Force'] = (
        uk_map['PFANM']
        .str.lower()
        .str.replace(" police", "")
        .str.replace(" & ", " and ")
        .str.replace(" ", "-")      
    )

    map_data = uk_map.merge(corr_df, on="Police_Force", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Temperature vs. Crime Rate Correlation by Police Force", fontsize=16, fontweight='bold')

    # Plot the map using a red colormap
    map_data.plot(
        column="Temp_Crime_Correlation",
        cmap="Reds",
        linewidth=0.5,
        edgecolor='0.5',
        ax=ax,
        legend=True,
        legend_kwds={
            "label": "Pearson Correlation (Mean Temp vs Crime Rate)",
            "orientation": "horizontal",
            "shrink": 0.7,
            "pad": 0.05
        },
        missing_kwds={"color": "lightgrey",
        }
    )

    ax.axis('off')
    plt.tight_layout()
    output_img = "outputs/Presentation Temperature/temperature_correlation_map.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nImage saved as {output_img}")

if __name__ == "__main__":
    generate_temp_correlation_map()