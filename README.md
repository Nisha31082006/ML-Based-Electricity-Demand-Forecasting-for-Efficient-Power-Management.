## ⚡ Electricity Load Prediction using Machine Learning & Deep Learning
##  🔗Colab Link

https://colab.research.google.com/drive/1Q1n2PdmJfvIzGEtpME9vN6TNziB3aRli

## 📌 Project Overview

Electricity demand varies continuously based on time, weather conditions, and energy generation sources. Accurate prediction of electricity load is essential for efficient power generation planning, avoiding power shortages, and ensuring effective energy management.

This project uses Machine Learning and Deep Learning techniques to analyze historical energy and weather data, predict electricity demand, and classify load levels.

## 🎯 Objectives
Preprocess and integrate energy and weather datasets
Predict electricity load using regression models
Classify electricity load into Low, Medium, High categories
Compare multiple ML algorithms
Identify the best-performing model
Visualize predictions and performance metrics

## 📊 Dataset
Source: Kaggle
### Link: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

### Dataset Details
Type: Time-series tabular data
Records: ~35,000+
Features: ~10–15 selected features

### Features Used
Temperature
Humidity
Wind Speed
Cloud Coverage
Energy generation sources
Lag and rolling features

### Target Variables
Regression: total_load_actual
Classification: Load Category (Low / Medium / High)

## ⚙️ Data Preprocessing
Selected relevant features
Converted time to datetime
Merged energy and weather datasets
Removed missing values
Encoded categorical variables
Feature scaling using StandardScaler
Converted data to categorical (for Apriori)

## 🤖 Machine Learning Models
### 🔹 Regression
Linear Regression
Decision Tree Regressor
Random Forest Regressor
SVR
KNN Regressor

### 📈 Evaluation Metrics
Regression
R² Score
MAE
MSE
RMSE

### 🔹 Classification
Naive Bayes
Decision Tree
Random Forest ⭐ (Best Model)
SVM
KNN

### Classification
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

## 🧠 Deep Learning Model (ANN)
Input Layer
Hidden Layers (ReLU Activation)
Output Layer (Softmax)
Optimizer: Adam
Loss Function: Categorical Crossentropy
 Achieved ~94% accuracy

## 🔍 Clustering Techniques
K-Means (Elbow Method)
Hierarchical Clustering (Dendrogram)
DBSCAN  (Best based on metrics)
### Evaluation Metrics
Silhouette Score
Davies-Bouldin Index

## 🔗 Association Rule Mining (Apriori)
Generated frequent itemsets
Identified relationships between features
### Metrics used:
Support
Confidence
Lift
Used only for pattern analysis (not for prediction)

## 🧠 Recommendation System

A simple rule-based recommendation system:
High Load → Reduce electricity usage
Medium Load → Use efficiently
Low Load → Normal usage

## 🌐 Web Application
A user-friendly web application is developed using Streamlit.
### 🔹 Features
### User Inputs:
Temperature
Humidity
Wind Speed
### Outputs:
Predicted Load (Low / Medium / High)
Recommendation message
Provides real-time prediction using trained model

## 📊 Results
Random Forest achieved the highest performance
ANN achieved high accuracy (~94%)
Other models showed moderate performance

## 🧠 Machine Learning vs Deep Learning
ANN captures complex non-linear patterns
Random Forest performs better for tabular data
Final model selected: Random Forest

## ⚙️ Project Pipeline
Data Collection → Preprocessing → Feature Scaling → 
Clustering → ML Models → Deep Learning (ANN) → 
Model Comparison → Recommendation System → Web App

## ▶️ How to Run the Project
🔹 Clone Repository
git clone <your-repo-link>
🔹 Install Dependencies
pip install numpy pandas scikit-learn tensorflow mlxtend matplotlib streamlit
🔹 Run Notebook
Open .ipynb file in Jupyter Notebook or Google Colab
Run all cells
🔹 Run Web Application
streamlit run app.py

## 📦 Dependencies
Python
NumPy
Pandas
Scikit-learn
TensorFlow / Keras
Matplotlib
mlxtend
Streamlit

## 🏁 Final Conclusion
DBSCAN performed best for clustering
ANN showed strong deep learning performance
Apriori revealed useful patterns
Random Forest achieved the best overall accuracy
###  Final Model Selected: Random Forest

## 👩‍💻 Team Members
SRUTHI R
THARSINIJEYASIHA V
