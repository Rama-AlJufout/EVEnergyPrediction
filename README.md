<div align="center">

# EV Energy Prediction

### A data-driven way to understand and predict electric-vehicle energy consumption

<p>
  <img src="assets/ev-energy-prediction-cover.svg" alt="EV Energy Prediction banner" width="100%">
</p>

<p>
  <a href="https://github.com/Rama-AlJufout/EVEnergyPrediction"><img src="https://img.shields.io/badge/project-EV%20Energy%20Prediction-0b827b?style=for-the-badge" alt="Project"></a>
  <img src="https://img.shields.io/badge/task-regression-26d6c6?style=for-the-badge" alt="Regression">
  <img src="https://img.shields.io/badge/python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p>
  <strong>Predicting the energy an EV needs from the way it moves, the road it uses, the weather it faces, and the condition of its vehicle and battery.</strong>
</p>

</div>

---

## The project in one minute

This repository contains an end-to-end machine-learning workflow for predicting the continuous target `Energy_Consumption_kWh`.

The project moves from **data quality** to **EDA**, **feature engineering**, **leakage-aware preprocessing**, **model comparison**, and **held-out evaluation**. The final notebook selects **Linear Regression** and records a test **RMSE of 0.5056 kWh** and **R² of 0.9458**.

> **Important:** These metrics describe the dataset and reproducible split used in the notebook. They are not a promise of performance on a different fleet, vehicle, or operating environment.

### Final scorecard

<div align="center">

| Selected model | Test MAE | Test RMSE | Test R² |
|:---:|:---:|:---:|:---:|
| **Linear Regression** | **0.4070 kWh** | **0.5056 kWh** | **0.9458** |

</div>

## Quick navigation

<table>
<tr>
<td><a href="#business-problem-and-objectives">Problem &amp; objectives</a></td>
<td><a href="#models-and-evaluation">Results</a></td>
<td><a href="#the-workflow">Workflow</a></td>
<td><a href="#data-understanding">Data</a></td>
</tr>
<tr>
<td><a href="#repository-data-artifacts">Data files</a></td>
<td><a href="#technology-stack">Technology</a></td>
<td><a href="#run-the-project">Run it</a></td>
<td><a href="#limitations-and-next-steps">Next steps</a></td>
</tr>
</table>

## Business problem and objectives

EV energy use is affected by much more than distance. Speed, acceleration, slope, traffic, weather, battery state, tire pressure, and vehicle weight can all change the energy required for a trip.

This project predicts `Energy_Consumption_kWh` from those driving, road, traffic, weather, vehicle, and battery conditions. Its objectives are to:

- build and evaluate a supervised regression model for EV energy consumption;
- analyze how driving behavior and operating conditions relate to energy use;
- compare multiple regression algorithms using appropriate metrics;
- identify useful predictive features without introducing data leakage.

Practical applications include trip-energy planning, range awareness, efficient-driving analysis, vehicle and battery monitoring, and data-driven energy-management decisions. These applications require independent validation before operational use.

## Models and evaluation

### Validation model comparison

<div align="center">
<img src="assets/model-comparison.png" alt="Validation RMSE comparison across regression models" width="88%">
</div>

| Model | Validation MAE | Validation RMSE | Validation R² | Decision |
|---|---:|---:|---:|---|
| **Linear Regression** | **0.3945 kWh** | **0.4969 kWh** | **0.9516** | **Selected** |
| Random Forest | 0.5190 kWh | 0.6570 kWh | 0.9154 | Compared |
| Decision Tree | 0.7008 kWh | 0.8880 kWh | 0.8454 | Compared |
| KNN Regressor | 0.9050 kWh | 1.1350 kWh | 0.7474 | Compared |

### Five-fold cross-validation on the training partition

| Model | Mean RMSE | RMSE standard deviation | Mean MAE | Mean R² |
|---|---:|---:|---:|---:|
| **Linear Regression** | **0.5214 kWh** | **0.0100** | **0.4164 kWh** | **0.9425** |
| Random Forest | 0.6596 kWh | 0.0078 | 0.5273 kWh | 0.9080 |
| Decision Tree | 0.8661 kWh | 0.0139 | 0.6923 kWh | 0.8413 |
| KNN Regressor | 1.1106 kWh | 0.0107 | 0.8979 kWh | 0.7392 |

Linear Regression has the best validation and cross-validation results, so it is selected for the final held-out test evaluation.

### The strongest recorded linear associations

<div align="center">
<img src="assets/feature-correlations.png" alt="Feature correlations with energy consumption" width="88%">
</div>

- `Speed_kmh`: **0.793** positive correlation.
- `Distance_Travelled_km`: **0.466** positive correlation.
- `Battery_State_%`: **-0.214** negative correlation.

Correlation is useful for inspection, but it does **not** prove causation.

### The target has a measurable operating range

<div align="center">
<img src="assets/target-distribution.png" alt="Energy consumption target distribution" width="88%">
</div>

The complete dataset contains 5,000 observations. The observed target ranges from approximately **1.881 kWh to 14.661 kWh** in the main dataset.

## The workflow

The implementation is available in [`code/EV_Energy_Prediction.ipynb`](code/EV_Energy_Prediction.ipynb). Each stage has a documented action and a recorded result.

### 01 — Load and understand

The notebook imports the analysis stack, loads the CSV, and parses `Timestamp` as datetime.

**Result:** The modeling input starts at **5,000 rows × 19 columns**. Because `Energy_Consumption_kWh` is continuous, the task is framed as supervised regression.

### 02 — Audit and clean

The workflow creates a working copy, checks exact duplicates, validates timestamps, and sorts observations chronologically.

**Result:** The main working table remains **5,000 × 19** with no invalid timestamps removed. The separate dirty file demonstrates why these checks matter: it contains **5,025 rows, 24 duplicate rows, and 5,038 missing cells**, along with implausible extremes.

### 03 — Define the prediction problem

`Energy_Consumption_kWh` is separated as the target. `Driving_Mode`, `Road_Type`, `Traffic_Condition`, and `Weather_Condition` are handled as categorical features because their numeric values are category codes, not measurements.

**Result:** The model does not assume that category code `3` is mathematically greater than category code `1`.

### 04 — Explore the data

The notebook generates a target histogram, categorical box plots, numerical scatter plots, and correlation analysis.

**Result:** The charts expose target spread and feature associations while the interpretation stays careful: an association is not a causal explanation.

### 05 — Inspect outliers

The IQR rule is applied to numerical features and the target:

```text
lower bound = Q1 - 1.5 × IQR
upper bound = Q3 + 1.5 × IQR
```

**Result:** The notebook reports **zero IQR outliers** in the inspected columns. This is an inspection result, not a reason to delete valid observations automatically.

### 06 — Engineer time features

The timestamp is transformed into:

- `Hour_of_Day`
- `Day_of_Week`

**Result:** **Zero timestamps failed to parse**, and the two time features were created successfully. The raw timestamp is removed from model inputs after its useful information is extracted.

### 07 — Build a leakage-aware feature matrix

The model excludes:

- `Energy_Consumption_kWh` — the target;
- `Vehicle_ID` — an identifier, not a vehicle condition;
- raw `Timestamp` — represented through engineered time features.

**Result:** The notebook reports **18 input features before preprocessing**.

### 08 — Split and preprocess safely

The data is split with `random_state=42`:

```text
70% training  → fit transformations and models
10% validation → compare candidate models
20% testing   → final evaluation only
```

Imputation, one-hot encoding, and scaling are fitted inside a scikit-learn `Pipeline` and `ColumnTransformer`.

**Result:** Validation and test information is not used to learn preprocessing parameters. The final test partition stays untouched until model selection is complete.

### 09 — Compare candidate models

The workflow compares Linear Regression, KNN Regressor, Decision Tree, and Random Forest. It also runs five-fold cross-validation on the training partition.

**Result:** Linear Regression achieves the lowest validation RMSE (**0.4969 kWh**) and is selected.

### 10 — Evaluate once on unseen data

The selected pipeline predicts the held-out test partition.

**Result:** The final evaluation is **MAE 0.4070 kWh**, **RMSE 0.5056 kWh**, and **R² 0.9458**. The close validation and test RMSE values provide a useful consistency check for this split.

## Data understanding

### Dataset schema

| Column | Description | Model role |
|---|---|---|
| `Vehicle_ID` | Vehicle identifier | Excluded |
| `Timestamp` | Observation date and time | Transformed |
| `Speed_kmh` | Speed in km/h | Numeric feature |
| `Acceleration_ms2` | Acceleration in m/s² | Numeric feature |
| `Battery_State_%` | Battery state percentage | Numeric feature |
| `Battery_Voltage_V` | Battery voltage | Numeric feature |
| `Battery_Temperature_C` | Battery temperature | Numeric feature |
| `Driving_Mode` | Encoded driving mode | Categorical feature |
| `Road_Type` | Encoded road type | Categorical feature |
| `Traffic_Condition` | Encoded traffic condition | Categorical feature |
| `Slope_%` | Road slope percentage | Numeric feature |
| `Weather_Condition` | Encoded weather category | Categorical feature |
| `Temperature_C` | Ambient temperature | Numeric feature |
| `Humidity_%` | Relative humidity | Numeric feature |
| `Wind_Speed_ms` | Wind speed in m/s | Numeric feature |
| `Tire_Pressure_psi` | Tire pressure in PSI | Numeric feature |
| `Vehicle_Weight_kg` | Vehicle weight in kg | Numeric feature |
| `Distance_Travelled_km` | Distance travelled in km | Numeric feature |
| `Energy_Consumption_kWh` | Energy consumed in kWh | **Target** |

### Repository data artifacts

| File | What it contains |
|---|---|
| [`Data/EV_Energy_Consumption_Dataset.csv`](Data/EV_Energy_Consumption_Dataset.csv) | 5,000-row complete dataset used by the main modeling workflow. |
| [`Data/data.csv`](Data/data.csv) | 5,000-row working copy with the same schema. |
| [`Data/EV_Energy_Consumption_Dirty.csv`](Data/EV_Energy_Consumption_Dirty.csv) | 5,025-row quality-audit input with missing values, duplicates, and implausible values. |
| [`Data/EV_Energy_Consumption_Cleaned.csv`](Data/EV_Energy_Consumption_Cleaned.csv) | 4,000-row cleaned export retained as a separate project artifact. |

The project is based on the [EV Energy Consumption Dataset on Kaggle](https://www.kaggle.com/datasets/ziya07/ev-energy-consumption-dataset).

## Technology stack

### Languages and formats

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Markdown-000000?style=flat-square&logo=markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/CSV-1D6F42?style=flat-square" alt="CSV">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/SVG-FFB13B?style=flat-square" alt="SVG">
</p>

### Python libraries

| Area | Packages |
|---|---|
| Data and numerical computing | `pandas`, `numpy`, `scipy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Persistence and explainability | `joblib`, `shap` |
| Notebook environment | `jupyter`, `ipykernel`, `nbformat` |

The main executed notebook directly uses pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. The full package set is declared in [`requirements.txt`](requirements.txt) for the wider project workflow.

## Project structure

```text
EVEnergyPrediction/
├── assets/
│   ├── ev-energy-prediction-cover.svg
│   ├── feature-correlations.png
│   ├── model-comparison.png
│   └── target-distribution.png
├── Data/
│   ├── data.csv
│   ├── EV_Energy_Consumption_Cleaned.csv
│   ├── EV_Energy_Consumption_Dataset.csv
│   ├── EV_Energy_Consumption_Dirty.csv
│   └── new_data.ipynb
├── code/
│   ├── EV_Energy_Prediction.ipynb
│   └── ev_energy_milestone_3.py
├── README.md
└── requirements.txt
```

## Run the project

### 1. Clone the repository

```bash
git clone https://github.com/Rama-AlJufout/EVEnergyPrediction.git
cd EVEnergyPrediction
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and launch Jupyter

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open [`code/EV_Energy_Prediction.ipynb`](code/EV_Energy_Prediction.ipynb) and run the cells from top to bottom.

## Limitations and next steps

- The recorded evaluation uses one reproducible random split; a chronological split would better simulate future-trip prediction.
- Category meanings should be confirmed against the original dataset documentation before operational use.
- Automated schema, range, missingness, and duplicate checks should be added for repeatable data validation.
- Repeated cross-validation and hyperparameter tuning can test how stable the model ranking is.
- A production version should save the fitted pipeline, validate prediction inputs, monitor drift, and report uncertainty.
- The final metrics should be tested on an independent real-world dataset before supporting vehicle or charging decisions.

## Project status

| Milestone | Status |
|---|:---:|
| Problem definition and dataset understanding | Complete |
| Data-quality audit and cleaning | Complete |
| EDA, correlations, and feature engineering | Complete |
| Leakage-aware preprocessing | Complete |
| Model comparison and validation | Complete |
| Held-out test evaluation | Complete |
| Production packaging and independent validation | Planned |

## Team

- Rama AlJufout
- Areen Ahmad AlJarrah
- Osama Nayfeh
- Abdallah Al-Daradkeh
- Ra'fat Osama Falah AlDiabat

## References

- [EV Energy Consumption Dataset — Kaggle](https://www.kaggle.com/datasets/ziya07/ev-energy-consumption-dataset)
- [Main project notebook](code/EV_Energy_Prediction.ipynb)
- [Python dependencies](requirements.txt)

---
