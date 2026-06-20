# Addressing real-world crime and security problems with data science.
This project was developed for the course 4CBLW020 by Group 11 to answer the following research question:

*“How can data‑driven estimates of police demand, using temperature and free time, inform more efficient allocation of patrol resources across LSOAs?”*

Police forces face seasonal demand spikes but current planning uses annual averages. The Police Allocation Formula ignore seasonality.
We developed two forecasting model - OCO (Online Convec Optimization) and SARIMAX to predict monthly crime at LSOA level using temperature, free time and past crime. The models are compared against a naive seasonal baseline and integrated into an interactive Streamlit dashboard.

## Requirements:
- Git
- Python 3.11.9+

## Installation and Setup:

### Clone the repository
```bash
git clone https://github.com/Ayleks47/4CBLW020-Group-11.git
cd 4CBLW020-Group-11
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download the Data

Find the data in this [Google Drive]()

### Run the Project

There are 5 folders in this project. The order in which they are run matters. To run each part of the project:


**(1) src/data_engineering**

Data transformations.

```bash
python src/data_engineering/ ???????????
```


**(2) src/visualisation**

Generate various graphs and figures related to the project.

```bash
# Replace "script" with the name of any visualisation script.
python src/visualisation/"script".py
```


**(3) src/SARIMAX**

SARIMAX model.
```bash
python src/SARIMAX/sarimax.py
```


**(4) src/oco**

OCO Model.
```bash
# Setup.
python src/oco/oco_data.py

# Main model.
python src/oco/oco_model.py

# OCO vs Naive comparison.
python src/oco/naive.py
```


**(5) src/dashboard**

This is the main dashboard (ensure you run data engineering and models before).
```bash
# Replace folder and script with the specific folder and script you want to run.
python src/folder/script.py
```


## Important Files to Know (Shared in google drive)
data/final_midterm_prototype_with_rates.csv -> This is our Master Dataset. It contains Month, Police Force, Mean Temp, Rainfall, Sunshine, Population, and Crime Rate. We will use this to train our predictive model for the dashboard. After obtaining tourism levels and free days, we will join the dataset.

outputs/Presentation Temperature/temperature_correlation_map.png -> The Red Map that shows how each police force area is affected by the change in temperature. If the Police Force area is in dark red, it means that it is highly correlated to the temperature.

outputs/Presentation Temperature/national_time_series_correlation.png -> The dual axis line chart. This essentially is a finding and a proof of feasibility for the subquestion of "Does temperature help explain fluctuations in selected crime types?". Where we see there is a big correlation.

## Presentation Talking Points
Our specific sub-question is: "Does temperature help explain fluctuations in selected crime types?"

We can answer this with:

The Temporal Proof (Line Chart): Show the 5-year dual-axis chart. Explain that the national crime rate moves in perfect synchronization with the national temperature. Every summer it peaks; every winter it drops.

The Spatial Proof (Red Map): Show the correlation heatmap. Explain that we calculated the r (Pearson) correlation for all 43 forces. This proves the effect isn't uniform—coastal and tourist regions (Dark Red) are hyper-sensitive to heat compared to dense urban centers.

The Criminological Theory: We explain this using Routine Activity Theory (1979, Cohen & Felson). Heat doesn't biologically cause crime; it pushes people outdoors, floods tourist areas, and leaves homes empty with windows open. It mathematically increases "Suitable Targets" and decreases "Capable Guardians."

"Routine Activity Theory (Cohen & Felson, 1979) - Google it if you want to learn more about it.
For a crime to happen, you need an offender and a target in the same place at the same time. Weather dictates human routine. Rain keeps people indoors (lowering street crime, but maybe increasing domestic incidents). Sun brings people to parks and pubs (increasing public order offenses)" - This is the relevant literature/theory for this sub-problem. You could also tie this to tourism, as more hot weather, more tourists come etc.

## Feasibility for Final End Goal
We present some of these plots as our Exploratory Data Analysis (EDA). By proving that temperature heavily correlates with crime, we prove to the examiners that our ultimate goal, an AI-driven Interactive Dashboard that forecasts crime based on upcoming weather reports, tourism and free-time is feasible and highly valuable for police resource allocation.

## Contributors
- Mariana Vall (2133768)
- Kerem Yildiz ()
- Alex Vančo ()
- Maciej Śliżak ()
- Vlad Ionescu ()
- Yağmur Çalişkan ()
