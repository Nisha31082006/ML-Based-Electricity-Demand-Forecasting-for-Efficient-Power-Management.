# ⚡ Electricity Load Analysis and Prediction using Machine Learning

https://colab.research.google.com/drive/1Q1n2PdmJfvIzGEtpME9vN6TNziB3aRli

---

## 📌 Project Overview
Electricity demand varies continuously based on time, weather conditions, and energy generation sources. Accurate prediction of electricity load is essential for efficient power generation planning, avoiding power shortages, and ensuring effective energy management.

This project uses **Machine Learning and Deep Learning techniques** to analyze historical energy and weather data, predict electricity demand, and classify load levels.

---

## 🎯 Objectives
- Preprocess and integrate energy and weather datasets  
- Predict electricity load using regression models  
- Classify electricity load into Low, Medium, High categories  
- Compare multiple ML algorithms  
- Identify the best-performing model  

---

## 📊 Dataset
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather  

### Dataset Details
- Type: Time-series tabular data  
- Records: ~35,000+  
- Features: ~10–15 selected features  

### Features Used
- Temperature  
- Humidity  
- Wind Speed  
- Cloud Coverage  
- Energy generation sources  
- Lag and rolling features  

### Target Variables
- Regression: `total_load_actual`  
- Classification: Load Category (Low / Medium / High)

---

## ⚙️ Data Preprocessing
- Selected relevant features  
- Converted time to datetime  
- Merged energy and weather datasets  
- Handled missing values  
- Created lag features  
- Applied normalization  

---

## 🤖 Models Implemented

### 🔹 Regression
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- SVR  
- KNN Regressor  

### 🔹 Classification
- Naive Bayes  
- Decision Tree  
- Random Forest  
- SVM  
- KNN  

### 🔹 Deep Learning
- Artificial Neural Network (ANN)

---

## 📈 Evaluation Metrics

### Regression
- R² Score  
- MAE  
- MSE  
- RMSE  

### Classification
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 📊 Results
- Random Forest achieved the highest performance  
- ANN performed well but required more tuning  
- Other models showed moderate performance  

---

## 🧠 Deep Learning vs Machine Learning
- ANN captures complex patterns  
- Random Forest performed better for tabular data  
- Final model selected: **Random Forest**

---

## 🔗 Association Rule Mining
- Apriori algorithm used for pattern discovery  
- Found relationships between weather and electricity load  

> Note: Apriori is not used for prediction, only for pattern analysis.

---

## 🏆 Conclusion
- Random Forest is the best-performing model  
- Provides high accuracy and stability  
- Suitable for electricity load prediction  

---



## 👩‍💻 Team Members
SRUTHI R ,
THARSINIJEYASIHA V



---
