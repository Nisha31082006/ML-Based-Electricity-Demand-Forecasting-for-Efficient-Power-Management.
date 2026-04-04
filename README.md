# ⚡ Electricity Load Analysis and Prediction using Machine Learning

## 📌 Project Overview
Electricity demand varies continuously based on time, weather conditions, and energy generation sources. Accurate prediction of electricity load is essential for efficient power generation planning, avoiding power shortages, and ensuring effective energy management.

This project uses **Machine Learning and Deep Learning techniques** to analyze historical energy and weather data, predict electricity demand, and classify load levels.

---

## 🎯 Objectives
- Preprocess and integrate energy and weather datasets  
- Predict electricity load using regression models  
- Classify electricity load into Low, Medium, High categories  
- Implement and compare multiple ML algorithms  
- Apply different train-test splitting methods  
- Identify the best-performing model  

---

## 📊 Dataset
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

### Dataset Description
- Type: Time-series tabular data  
- Records: ~35,000+  
- Features: ~10–15 selected features  

### Features Used
- Temperature  
- Humidity  
- Wind Speed  
- Cloud Coverage  
- Energy generation sources  
- Lag features and rolling averages  

### Target Variables
- Regression: `total_load_actual`  
- Classification: Load Category (Low / Medium / High)

---

## ⚙️ Data Preprocessing
- Selected relevant features  
- Converted time column to datetime  
- Merged energy and weather datasets  
- Handled missing values (forward fill)  
- Created lag features (lag-1, lag-24)  
- Applied normalization (Min-Max Scaling)  

---

## 🤖 Models Implemented

### 🔹 Regression Models
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Support Vector Regression (SVR)  
- KNN Regressor  

### 🔹 Classification Models
- Naive Bayes  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbours (KNN)  

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

## 🔍 Train-Test Splitting Methods

### Regression
- Manual Time-Series Split  
- Percentage Split  
- TimeSeriesSplit (Best)

### Classification
- Random Split  
- Stratified Split  
- K-Fold  
- Stratified K-Fold  

---

## 📊 Results & Comparison
- Random Forest achieved the highest performance among all models  
- ANN performed well but required more tuning  
- Decision Tree and KNN showed moderate performance  
- SVM and Naive Bayes gave comparatively lower performance  

---

## 🧠 Deep Learning vs Machine Learning
- ANN used for capturing complex patterns  
- Random Forest outperformed ANN for tabular data  
- Final model selected: **Random Forest**

---

## 🔗 Association Rule Mining
- Apriori algorithm used for pattern discovery  
- Identified relationships between weather and electricity load  

### Note:
Apriori is not used for prediction; it is used only for finding patterns.

---

## 🏆 Final Conclusion
- Random Forest is the best model for this dataset  
- Provides highest accuracy and stability  
- Suitable for tabular data and reduces overfitting  
- ANN used for comparison but not selected as final model  

---

## 🚀 Future Enhancements
- Implement LSTM for time-series forecasting  
- Add more features (holidays, seasonal data)  
- Improve classification categories  
- Deploy as web application or dashboard  

---

## 📌 Applications
- Power grid management  
- Electricity demand monitoring  
- Energy distribution planning  
- Smart grid systems  

---

## 👩‍💻 Team Members
SRUTHI R 
THARSINIJEYASIHA 



---
