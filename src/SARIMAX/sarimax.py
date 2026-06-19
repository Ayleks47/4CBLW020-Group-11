import argparse
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

"""
Run in cmd in the directory where the file is:
python sarimax.py --data "path here" -n_jobs -1 (or however many threads you wanna use, -1 means using all available)
"""

# Data dependent variables
SEASON_LEN = 12 
FREQ = "MS"
EXOG_COL = "Mean_Temperature_C"

#Data loading
def load_data(path):
    df = pl.read_parquet(path)
    
    return (
        df.with_columns([
            pl.col("LSOA code").cast(pl.Utf8),
            pl.col("Month").cast(pl.Date),
            pl.col("Crime_Count").cast(pl.Float32),
            pl.col(EXOG_COL).cast(pl.Float32),
        ])
        .group_by(["LSOA code", "Month"])
        .agg([
            pl.col("Crime_Count").sum().alias("Total_Crime_Count"),
            pl.col(EXOG_COL).mean().alias(EXOG_COL),
        ])
        .sort(["LSOA code", "Month"])
    )

def separate_training(df, holdout):
    sorted_months = df["Month"].unique().sort()
    test_months = sorted_months.tail(holdout)
    train = df.filter(~pl.col("Month").is_in(test_months))
    test  = df.filter(pl.col("Month").is_in(test_months))
    return train, test

def convert_format(df):
    base = (
        df.rename({
            "LSOA code": "unique_id",
            "Month": "ds",
            "Total_Crime_Count": "y",
        })
        .sort(["unique_id", "ds"])
        .to_pandas()
        .assign(
            unique_id=lambda x: x["unique_id"].astype(str),
            ds=lambda x: pd.to_datetime(x["ds"]),
            y=lambda x: x["y"].astype(np.float32),
        )
    )

    y_df = base[["unique_id", "ds", "y", EXOG_COL]].copy() 
    exog_df = base[["unique_id", "ds", EXOG_COL]].copy()
    exog_df[EXOG_COL] = exog_df[EXOG_COL].astype(np.float32)

    return y_df, exog_df

def aggregate_temps(train_exog):
    return (
        train_exog.assign(month_of_year=lambda x: x["ds"].dt.month)
        .groupby(["unique_id", "month_of_year"], as_index=False)[EXOG_COL]
        .mean()
        .sort_values(["unique_id", "month_of_year"])
        .reset_index(drop=True)
    )

# Model fitting
def build_sf(n_jobs):
    return StatsForecast(
        models=[
            AutoARIMA(
                season_length=SEASON_LEN,
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                max_d=2,
                max_D=1,
                nmodels=30,
                approximation=True,
                stepwise=True,
                allowdrift=False
            )
        ],
        freq=FREQ,
        n_jobs=n_jobs,
    )

# Forecasting and metrics evaluation
def select_forecast(fcst_raw):
    df = fcst_raw.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    
    return df[["unique_id", "ds", "AutoARIMA"]].rename(columns={"AutoARIMA": "forecast"})

def evaluate(forecasts, test):
    actual = (
        test.rename({
            "LSOA code": "unique_id",
            "Month": "ds",
            "Total_Crime_Count": "y",
        })
        .select(["unique_id", "ds", "y"])
        .to_pandas()
        .assign(ds=lambda x: pd.to_datetime(x["ds"]))
    )

    merged = forecasts.merge(actual, on=["unique_id", "ds"], how="inner")
    err = merged["forecast"] - merged["y"]

    per_lsoa = (
        merged.assign(ae=err.abs(), se=err ** 2)
        .groupby("unique_id", as_index=False)
        .agg(
            n_months=("y", "count"),
            MAE=("ae", "mean"),
            RMSE=("se", lambda s: float(np.sqrt(s.mean()))),
        )
        .sort_values("MAE", ascending=False)
    )

    return pl.from_pandas(per_lsoa)

# Pickle model and save weather data and forecasting dates
def save_model(sf, train_y, seasonal_temps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "sf_sarimax.pkl"

    with open(model_path, "wb") as fh:
        pickle.dump(sf, fh, protocol=5)

    st_path = out_dir / "seasonal_temps.parquet"
    pl.from_pandas(seasonal_temps).write_parquet(st_path)

    last_dates = (
        train_y.groupby("unique_id", as_index=False)["ds"]
        .max()
        .rename(columns={"ds": "last_ds"})
    )
    ld_path = out_dir / "last_train_dates.parquet"
    pl.from_pandas(last_dates).write_parquet(ld_path)

    return model_path

def main(data_path, holdout, n_jobs, out_dir):
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("data Loading - 1")
    df = load_data(data_path)

    print(" - 2")
    train_df, test_df = separate_training(df, holdout)

    print(" - 3")
    train_y, train_exog = convert_format(train_df)
    test_y, test_exog = convert_format(test_df)

    print(" - 4")
    seasonal_temps = aggregate_temps(train_exog)

    print("model fitting")
    sf = build_sf(n_jobs=n_jobs)
    sf.fit(train_y)

    print("foreacast")
    forecast_raw = sf.predict(h=holdout, X_df=test_exog)
    forecasts = select_forecast(forecast_raw)

    print("metrics")
    test_pl = pl.from_pandas(test_y.rename(columns={"unique_id": "LSOA code", "ds": "Month", "y": "Total_Crime_Count"}))
    metrics = evaluate(forecasts, test_pl)
    

    print("saving")
    save_model(sf, train_y, seasonal_temps, out_dir)
    metrics.write_parquet(out_dir / "metrics.parquet")
    pl.from_pandas(forecasts).write_parquet(out_dir / "test_forecasts.parquet")

    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=Path("lsoa_models"))
    args = parser.parse_args()

    main(args.data, args.holdout, args.n_jobs, args.out)
