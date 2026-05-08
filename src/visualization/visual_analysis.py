import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 2010-2026
df = pl.read_parquet("./merged_crime_dataset.parquet")

# List of crime types
crime_type = pl.Series(df.select('Crime type').unique()).to_list()

"""
Annotates the maximum in the plot (https://stackoverflow.com/questions/43374920/how-to-automatically-annotate-maximum-value-in-pyplot)
EDITED
"""
def annot_max(x,y, ax):
    if y is None or len(y) == 0:
        xmax = x[0] if (x is not None and len(x) > 0) else 0
        ymax = 0
    else:
        idx = np.argmax(y)
        xmax = x[idx]
        ymax = y[idx]
    # text= "y={:.3f}, x={f}".format(ymax, str(xmax))
    text = f"y={ymax}, x={xmax}"
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data',textcoords="axes fraction",
              bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

"""
Return plots of a specific crime in years 2012-2026, normalized
"""
def crime_type_figure(crime):
    year = 2012
    print(f'Crime type: {crime}\n')

    # Plotting crime 2010-2026
    fig, ax = plt.subplots(7,2,figsize=(10, 10))
    for i in range(7):
        for j in range(2):


            filtered_df = (df.filter(
                pl.col("Crime type") == crime,
                pl.col("Month").str.contains(str(year))
            ).select(
                pl.col("Month").unique(maintain_order=True).alias("unique"),
                pl.col("Month").unique_counts().alias("unique_counts"),
            )
            # normalization
            # .with_columns([
            # ((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())).alias(c)
            # for c in ["unique_counts"]
            # ])
            #
            )

            ax[i][j].plot(
                filtered_df["unique"],
                filtered_df["unique_counts"],
            )
            ax[i][j].set_title(year)
            ax[i][j].axes.get_xaxis().set_visible(False)
            annot_max(pl.Series(filtered_df.select('unique').unique()).to_list(), pl.Series(filtered_df.select('unique_counts').unique()).to_list(),ax[i][j])
            if year != 2026:
                year += 1
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(crime)
    plt.show()

"""
Return plots of all crimes in a given year, normalized
"""
def year_figure(year):

    print(f'Year: {year}\n')

    # Plotting every crime in "year"

    fig, ax = plt.subplots(int((len(crime_type)/2)),2,figsize=(10, 10))
    i = 0
    j = 0
    for crime in crime_type:
        filtered_df = df.filter(
            pl.col("Crime type") == crime,
            pl.col("Month").str.contains(str(year))
        ).select(
            pl.col("Month").unique(maintain_order=True).alias("unique"),
            pl.col("Month").unique_counts().alias("unique_counts"),
        ).with_columns([
        ((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())).alias(c) #normalization
        for c in ["unique_counts"]
        ])
        # print(filtered_df)
        # print(filtered_df.head())
        ax[i][j].plot(
            filtered_df["unique"],
            filtered_df["unique_counts"],
            # label = crime
        )
        ax[i][j].set_title(crime)
        ax[i][j].axes.get_xaxis().set_visible(False)
        if i < (len(crime_type)//2) - 1:
            i += 1
        else:
            i = 0
            j += 1
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(year)
    plt.tight_layout()
    plt.show()

for c in crime_type:
    crime_type_figure(c)
# for i in range(2010, 2026):
#     year_figure(i)

# print(df.filter((pl.col("Crime type") == "Public disorder and weapons")))

# Exclude 2010,2011,2026 for insufficient data