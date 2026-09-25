# ⚡ EV Energy Prediction

### Electric Vehicle Energy Consumption Prediction

**Team Members:**  
- Rama AlJufout
- Areen AlJarrah
- Osama Nayfeh
- Abdallah AlDaradkeh
- Ra'fat AlDiabat

---

## 1. 📌 Business Problem

Electric vehicle energy consumption is affected by many real-world factors, including driving behavior, road and traffic conditions, weather, and battery status. Accurately estimating energy consumption can help EV users and transportation systems better understand energy usage and support more efficient vehicle operation.

**EV Energy Prediction** aims to use Machine Learning to predict the energy consumption of an electric vehicle based on these real-world conditions.

---

## 2. 🎯 Project Objectives

The main objectives of EV Energy Prediction are:

- Build a Machine Learning model to predict **EV Energy Consumption (kWh)**.
- Analyze how driving behavior affects energy consumption.
- Study the impact of road, traffic, and slope conditions.
- Investigate the relationship between weather conditions and EV energy usage.
- Analyze the effect of battery and vehicle conditions on energy consumption.
- Compare different Regression Machine Learning models.
- Evaluate model performance using appropriate regression metrics.
- Identify the most influential features affecting energy consumption.

### What Success Looks Like

The project will be considered successful if the developed model can:

- Produce accurate energy consumption predictions.
- Generalize well to unseen data.
- Achieve acceptable **MAE, RMSE, and R²** scores.
- Provide useful insights into the factors that influence EV energy consumption.

---

## 3. 📊 Dataset Source & Overview

The dataset used in this project is the **EV Energy Consumption Dataset** available on Kaggle.

**Dataset Source:**  
[Kaggle - EV Energy Consumption Dataset](https://www.kaggle.com/datasets/ziya07/ev-energy-consumption-dataset)

The dataset contains parameters related to:

- Driving behavior
- Road conditions
- Traffic conditions
- Weather conditions
- Vehicle and battery characteristics

> **Dataset Size:** To be confirmed after the final dataset file is analyzed.

---

## 4. 🎯 Target Variable

The target variable for the Machine Learning model is:

`Energy_Consumption_kWh`

It represents the estimated electrical energy consumed by the electric vehicle, measured in **kilowatt-hours (kWh)**.

This makes the project a:

**Supervised Learning → Regression Problem**

The model will learn the relationship between the input features and energy consumption to predict the expected energy usage under different conditions.

---

# 📁 Dataset Description

The dataset provides detailed information about factors that can affect energy consumption in electric vehicles.

The available information can be divided into several main categories.

## 🚗 Driving Parameters

- **Speed (`Speed_kmh`)** — Vehicle speed measured in kilometers per hour.
- **Acceleration (`Acceleration_ms2`)** — Vehicle acceleration measured in meters per second squared.
- **Driving Mode (`Driving_Mode`)** — Selected driving mode, such as Eco, Normal, or Sport.

## 🛣️ Road & Traffic Conditions

- **Road Type (`Road_Type`)** — Type of road, such as Highway, Urban, or Rural.
- **Traffic Condition (`Traffic_Condition`)** — Traffic level affecting the vehicle.
- **Slope (`Slope_%`)** — Road slope percentage.
- **Distance Travelled (`Distance_Travelled_km`)** — Distance travelled by the vehicle in kilometers.

## 🌦️ Weather Factors

- **Temperature (`Temperature_C`)** — Ambient temperature in Celsius.
- **Humidity (`Humidity_%`)** — Relative humidity percentage.
- **Wind Speed (`Wind_Speed_ms`)** — Wind speed measured in meters per second.
- **Weather Condition (`Weather_Condition`)** — Weather conditions such as Sunny, Rainy, Snowy, or Foggy.

## 🔋 Vehicle & Battery Attributes

- **Battery State (`Battery_State_%`)** — Current battery state of charge (%).
- **Battery Voltage (`Battery_Voltage_V`)** — Battery voltage measured in volts.
- **Battery Temperature (`Battery_Temperature_C`)** — Battery temperature measured in Celsius.
- **Tire Pressure (`Tire_Pressure_psi`)** — Tire pressure measured in PSI.
- **Vehicle Weight** — Vehicle weight, if available in the final dataset and selected for modeling.

---

# 🔑 Key Features

For the initial project, we plan to focus on approximately **10–15 input features** rather than using every available column.

| Category | Features |
|---|---|
| 🚗 Driving | Speed, Acceleration, Driving Mode |
| 🛣️ Road & Traffic | Road Type, Traffic Condition, Slope, Distance Travelled |
| 🌦️ Weather | Temperature, Humidity, Wind Speed, Weather Condition |
| 🔋 Vehicle & Battery | Battery State, Battery Voltage, Battery Temperature, Tire Pressure |
| 🎯 Target | Energy Consumption (kWh) |

Final feature selection will be performed after data exploration and preprocessing.

---

# 🤖 Machine Learning Approach

EV Energy Prediction is formulated as a **Supervised Learning Regression** problem.

**Input:**

Driving Conditions + Road & Traffic Conditions + Weather Conditions + Vehicle & Battery Conditions

**Output:**

Predicted **Energy Consumption (kWh)**

---

# 💡 Example Use Case

A user provides the current vehicle and environmental conditions:

```text
Speed:               90 km/h
Acceleration:        1.2 m/s²
Battery State:       70%
Battery Voltage:     400 V
Battery Temperature: 30°C
Driving Mode:        Normal
Road Type:           Highway
Traffic Condition:   Low
Slope:               2%
Weather Condition:   Sunny
Temperature:         25°C
Humidity:            45%
Wind Speed:          4 m/s
Tire Pressure:       35 PSI
Distance Travelled:  50 km
```

The trained model then produces:

```text
Predicted Energy Consumption:
XX.XX kWh
```

---

# 🌱 Project Value

EV Energy Prediction can provide useful insights into the factors that influence EV energy consumption.

Potential applications include:

- ⚡ Energy consumption prediction
- 🚗 Understanding efficient driving behavior
- 🔋 Better EV energy management
- 📊 Data-driven vehicle performance analysis
- 🌱 Supporting more sustainable transportation

---

# 🛠️ Technologies

- **Python**
- **Jupyter Notebook**

Additional Machine Learning libraries may be used depending on the final model.

---

# 📂 Project Structure

```text
EV Energy Prediction/
│
├── data/
│   └── EV_Energy_Consumption.csv
│
├── README.md
└── requirements.txt
```

---

# 🚧 Project Status

**Status: In Development**

Current focus:

- Dataset understanding
- Data preprocessing
- Exploratory Data Analysis
- Feature selection

Results and final model performance will be added after the Machine Learning experiments are completed.
