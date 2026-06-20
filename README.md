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

## Contributors
- Mariana Vall (2133768)
- Kerem Yildiz ()
- Alex Vančo ()
- Maciej Śliżak ()
- Vlad Ionescu ()
- Yağmur Çalişkan (2083914)
