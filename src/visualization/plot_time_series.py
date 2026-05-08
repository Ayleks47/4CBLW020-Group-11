import pandas as pd
import matplotlib.pyplot as plt

def plot_dual_axis_timeseries():
    df = pd.read_csv("data/final_midterm_prototype_with_rates.csv")
    
    df = df.dropna(subset=['Crime_Rate_Per_1000', 'Mean_Temp'])
    
    # Take a 5 year window, avoid covid
    df = df[(df['Year'] >= 2015) & (df['Year'] <= 2019)]
    
    # Monthly
    national_trend = df.groupby('Month').agg({'Mean_Temp': 'mean','Crime_Rate_Per_1000': 'mean'}).reset_index()

    # Calculate the correlation
    correlation = national_trend['Mean_Temp'].corr(national_trend['Crime_Rate_Per_1000'])
    print(f"\nNational Correlation Score: {correlation:.2f}")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    months = national_trend['Month']
    
    # Plot temperature 
    color = 'tab:red'
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Mean Temperature (°C)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(months, national_trend['Mean_Temp'], color=color, linewidth=2.5, marker='o', label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Rotate X labels
    plt.xticks(months[::3], rotation=45) 

    # Create a second Y-axis that shares the same X-axis
    ax2 = ax1.twinx()  
    
    # Plot crime Rate 
    color = 'tab:purple'
    ax2.set_ylabel('Crime Rate (per 1,000 people)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(months, national_trend['Crime_Rate_Per_1000'], color=color, linewidth=2.5, marker='o', label='Crime Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title
    plt.title(f"National Correlation: Temperature vs. Crime Rate (2015-2019)\nPearson Correlation: {correlation:.2f}", fontsize=16, fontweight='bold')
    
    fig.tight_layout()
    output_img = "national_time_series_correlation.png"
    plt.savefig(output_img, dpi=300)
    print(f"Image saved as {output_img}")

if __name__ == "__main__":
    plot_dual_axis_timeseries()