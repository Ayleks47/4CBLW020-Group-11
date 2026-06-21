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

Find the data in this [OneDrive](https://tuenl-my.sharepoint.com/:f:/g/personal/a_vanco_student_tue_nl/IgBGOT3Ay212Q57eLknrh67bAcVgen3b9_qGxS-6ugcUwSE?e=kcscll), create a 'data' folder in the root folder and move all the data files into it.

### Run the Project

There are 5 folders in this project. The order in which they are run matters. To run each part of the project:


**(1) src/data_engineering**

Data transformations.

```bash
python src/data_engineering/add_populcation_data.py
python src/data_engineering/build_matrix.py
python src/data_engineering/build_prototype_data.py
python src/data_engineering/df.py
python src/data_engineering/extract_weather.py
python src/data_engineering/merge_weather.py
python src/data_engineering/prep_data.py
```


**(2) src/visualisation**

Generate various graphs and figures related to the project.

```bash
python src/visualisation/main_visualization.py
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
python src/dashboard/'name_of_script'.py
```


## Important Files to Know 


## Contributors
- Mariana Vall (2133768)
- Kerem Yildiz (2107872)
- Alex Vančo (2103265)
- Maciej Śliżak (2130076)
- Vlad Ionescu (2160692)
- Yağmur Çalişkan (2083914)
