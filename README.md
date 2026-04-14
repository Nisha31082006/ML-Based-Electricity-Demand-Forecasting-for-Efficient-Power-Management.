# ⚡ Smart Electricity Demand Prediction System

🚀 Built using Machine Learning + Deep Learning + Streamlit



## 🔗 Live Demo

⚠️ Currently running locally
Run using:

```bash
streamlit run app.py
```

## 🔗 Colab Notebook

https://colab.research.google.com/drive/1Q1n2PdmJfvIzGEtpME9vN6TNziB3aRli



## 📌 Project Overview

Electricity demand varies continuously based on weather conditions, time, and energy generation sources. Accurate prediction of electricity load is essential for efficient power management and avoiding power shortages.

This project uses **Machine Learning and Deep Learning techniques** to:

* Predict electricity demand
* Classify load levels (Low / Medium / High)
* Provide smart recommendations
* Build a real-time prediction web application



## 🎯 Objectives

* Preprocess and merge energy & weather datasets
* Predict electricity load using regression
* Classify electricity demand into categories
* Compare multiple ML models
* Identify the best-performing model
* Develop a user-friendly web application



## 📊 Dataset

**Source:** Kaggle
https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

### Dataset Details

* Type: Time-series tabular data
* Records: ~35,000+
* Features: Weather + energy generation

### Features Used

* Temperature
* Humidity
* Wind Speed
* Energy-related attributes

### Target Variables

* Regression → `total_load_actual`
* Classification → Load Category (Low / Medium / High)



## ⚙️ Data Preprocessing

* Datetime conversion
* Dataset merging (Energy + Weather)
* Handling missing values
* Feature selection
* Feature scaling
* Creation of categorical labels



## 🤖 Machine Learning Models

### 🔹 Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor ⭐ (Best)
* SVR
* KNN Regressor

### 🔹 Classification

* Naive Bayes
* Decision Tree
* Random Forest ⭐ (Best)
* SVM
* KNN

### 📈 Evaluation Metrics

* Regression → R² Score, MAE, RMSE
* Classification → Accuracy, Precision, Recall, F1-score



## 🧠 Deep Learning Model (ANN)

* Input Layer
* Hidden Layers (ReLU Activation)
* Output Layer (Softmax)
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Achieved ~94% accuracy



## 🔍 Clustering Techniques

* K-Means
* Hierarchical Clustering
* DBSCAN ⭐ (Best)


## 🔗 Association Rule Mining (Apriori)

* Generated frequent itemsets
* Discovered relationships between features

### Metrics Used

* Support
* Confidence
* Lift



## 🧠 Recommendation System

A rule-based recommendation system is implemented:

* High Load → Reduce electricity usage
* Medium Load → Use efficiently
* Low Load → Best time for heavy usage



## 🌐 Web Application (Streamlit)

### 🔹 Features

* Regression Prediction
* Classification Prediction
* ANN Prediction
* Apriori Logic
* Real-time Weather API integration
* 📊 Graph Visualization
* 📍 Prediction History
* 📥 Download Report
* 📈 Model Comparison
* 🏆 Best Model Identification



## 📊 Results

* Random Forest achieved the best performance
* ANN achieved high accuracy (~94%)
* Other models showed moderate performance



## 🧠 ML vs Deep Learning Insight

* Random Forest → Best for tabular data
* ANN → Captures complex non-linear patterns

👉 **Final Selected Model: Random Forest**



## ⚙️ Project Pipeline

Data Collection → Preprocessing → Feature Engineering →
Clustering → ML Models → ANN → Model Comparison →
Recommendation System → Web Application



## ▶️ How to Run the Project

### 🔹 Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 🔹 Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow mlxtend matplotlib streamlit
```

### 🔹 Run Application

```bash
streamlit run app.py
```



## 📦 Dependencies

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* mlxtend
* Streamlit



## 🏁 Final Conclusion

* DBSCAN performed best for clustering
* ANN showed strong deep learning performance
* Apriori revealed useful patterns
* Random Forest achieved the best overall performance

👉 **Final Model Selected: Random Forest**



## 📌 Note

Dataset files are not uploaded due to size limitations.
Please download from Kaggle and place in project folder before running.

---

## 👩‍💻 Team Members

* SRUTHI R
* THARSINIJEYASIHA V
