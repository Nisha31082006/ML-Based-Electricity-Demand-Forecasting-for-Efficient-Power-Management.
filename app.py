import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import datetime

from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Energy System", layout="wide")
st.title("⚡ Smart Electricity Demand System")

# =========================
# LOAD MODELS
# =========================
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
ann = load_model("ann_model.h5")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio("📌 Select Module", [
    "Regression",
    "Classification",
    "ANN",
    "Apriori",
    "History",
    "Model Comparison"
])

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# API
# =========================
API_KEY = "9804591c6d0b28e5cd3807041ce86c95"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200:
        st.error("City not found")
        return None

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"]
    }

# =========================
# RECOMMENDATION
# =========================
def recommend(temp, hum, wind, load):
    if load == "High":
        return "⚠️ Reduce AC usage & avoid heavy appliances"
    elif load == "Medium":
        return "🌤 Use electricity moderately"
    else:
        return "💡 Best time for heavy usage"

def personalized_tip(temp, hum, wind):
    if temp > 32:
        return "🔥 High temperature: Reduce AC usage"
    elif hum > 75:
        return "💧 Use fans instead of AC"
    elif wind > 10:
        return "💨 Use renewable energy"
    else:
        return "✅ Balanced weather"

def explain_prediction(temp, hum, wind):
    reasons = []
    if temp > 30:
        reasons.append("High temperature")
    if hum > 70:
        reasons.append("High humidity")
    if wind < 3:
        reasons.append("Low wind")
    return " + ".join(reasons) if reasons else "Normal conditions"

def demand_trend(category):
    if category == "High":
        return "Next hours: HIGH demand"
    elif category == "Medium":
        return "Stable demand"
    else:
        return "Low demand ahead"

# =========================
# INPUT
# =========================
city = st.text_input("Enter City Name")

# =========================
# PREDICTION
# =========================
if page in ["Regression","Classification","ANN","Apriori"]:

    if st.button("Predict Now"):

        weather = get_weather(city)

        if weather:

            temp, hum, wind = weather["temp"], weather["humidity"], weather["wind"]

            st.subheader("🌦 Weather Data")
            c1,c2,c3 = st.columns(3)
            c1.metric("Temp", f"{temp}°C")
            c2.metric("Humidity", f"{hum}%")
            c3.metric("Wind", f"{wind} m/s")

            # MODEL LOGIC
            if page == "Regression":
                pred = reg_model.predict([[temp, hum, wind]])[0]
                category = "Low" if pred<0.33 else "Medium" if pred<0.66 else "High"

            elif page == "Classification":
                category = clf_model.predict([[temp, hum, wind]])[0]

            elif page == "ANN":
                pred = np.argmax(ann.predict(np.array([[temp, hum, wind]])), axis=1)[0]
                category = ["Low","Medium","High"][pred]

            elif page == "Apriori":
                if temp>30 and hum>70:
                    category="High"
                elif temp>25:
                    category="Medium"
                else:
                    category="Low"

            # OUTPUT
            st.subheader("🔮 Prediction Result")

            if category=="High":
                st.error("⚡ High Demand 🚨")
                st.error("🚨 ALERT: Reduce usage immediately")
            elif category=="Medium":
                st.warning("⚡ Medium Demand")
            else:
                st.success("⚡ Low Demand")

            st.subheader("💡 Recommendation")
            st.write(recommend(temp, hum, wind, category))

            st.subheader("🎯 Personalized Tip")
            st.info(personalized_tip(temp, hum, wind))

            st.subheader("🧠 Explanation")
            st.write(explain_prediction(temp, hum, wind))

            st.subheader("📈 Future Trend")
            st.write(demand_trend(category))

            # ✅ ONLY BAR GRAPH (clean)
            st.subheader("📊 Graph")
            df_graph = pd.DataFrame({
                "Values":[temp, hum, wind]
            }, index=["Temp","Humidity","Wind"])

            st.bar_chart(df_graph)

            # Save history
            st.session_state.history.append({
                "Model":page,
                "City":city,
                "Temp":temp,
                "Humidity":hum,
                "Wind":wind,
                "Prediction":category
            })

# =========================
# HISTORY
# =========================
if page=="History":

    st.subheader("📍 History")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        st.download_button("Download Report",
                           df_hist.to_csv(index=False),
                           "history.csv")
    else:
        st.info("No history")

# =========================
# MODEL COMPARISON
# =========================
if page=="Model Comparison":

    st.subheader("📈 Model Comparison")

    energy = pd.read_csv("energy_dataset.csv")
    weather = pd.read_csv("weather_features.csv")

    weather = weather[['dt_iso','temp','humidity','wind_speed']]

    energy['time']=pd.to_datetime(energy['time'],utc=True)
    weather['dt_iso']=pd.to_datetime(weather['dt_iso'],utc=True)

    weather=weather.groupby('dt_iso').mean().reset_index()

    df=pd.merge(energy,weather,left_on='time',right_on='dt_iso',how='left')
    df.drop(columns=['dt_iso'],inplace=True)
    df.rename(columns={'total load actual':'total_load_actual'},inplace=True)
    df.ffill(inplace=True)

    df=df[['temp','humidity','wind_speed','total_load_actual']].dropna()

    X=df[['temp','humidity','wind_speed']]
    y_reg=df['total_load_actual']
    y_clf=pd.qcut(y_reg,3,labels=["Low","Medium","High"])

    X_train,X_test,y_train_reg,y_test_reg=train_test_split(X,y_reg,test_size=0.2,random_state=42)
    _,_,y_train_clf,y_test_clf=train_test_split(X,y_clf,test_size=0.2,random_state=42)

    reg_score=r2_score(y_test_reg,reg_model.predict(X_test))
    clf_score=accuracy_score(y_test_clf,clf_model.predict(X_test))
    ann_pred=np.argmax(ann.predict(X_test),axis=1)
    ann_score=accuracy_score(y_test_clf.cat.codes,ann_pred)

    scores={
        "Regression RF":round(reg_score,2),
        "Classification RF":round(clf_score,2),
        "ANN":round(ann_score,2)
    }

    df_models=pd.DataFrame(list(scores.items()),columns=["Model","Score"])
    st.dataframe(df_models)
    st.bar_chart(df_models.set_index("Model"))

    best=max(scores,key=scores.get)

    st.subheader("🏆 Best Model")
    st.success(f"{best} performs best with score {scores[best]}")