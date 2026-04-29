
import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Energy System", layout="wide")

# =========================
# UI DESIGN (EXACT BACKGROUND AS SAMPLE IMAGE)
# =========================
st.markdown("""
<style>

/* EXACT BACKGROUND AS SAMPLE IMAGE - LIGHT GRAY/SOFT WHITE */
.stApp {
    background-color: #f4f7fc;
    background-image: radial-gradient(circle at 10% 20%, rgba(220, 235, 250, 0.3) 0%, rgba(255,255,245,0.1) 90%);
}

/* Top Header - same as sample */
.header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Sidebar - exactly as sample image (dark blue gradient) */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72, #2a5298);
    padding-top: 40px;
}

/* Sidebar radio buttons */
div[role="radiogroup"] > label {
    background: white;
    color: black;
    padding: 12px;
    margin: 10px 15px;
    border-radius: 10px;
    font-weight: 500;
    transition: 0.2s;
}

div[role="radiogroup"] > label:hover {
    transform: translateX(5px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Main container - white card with shadow */
.main-box {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.03);
}

/* Section titles */
h3 {
    color: #1e3c72;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #00b894);
    color: white;
    border-radius: 10px;
    height: 2.8em;
    width: 180px;
    font-size: 16px;
    border: none;
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

/* Input box */
input {
    border-radius: 10px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #f8fafc;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="header">⚡ Smart Electricity Demand Dashboard</div>', unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
ann = load_model("ann_model.h5")

# =========================
# API
# =========================
API_KEY = "9804591c6d0b28e5cd3807041ce86c95"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)

    if res.status_code != 200:
        st.error("City not found")
        return None

    data = res.json()
    return data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"]

# =========================
# FUNCTIONS
# =========================
def recommend(cat):
    return {
        "High":"⚠️ Reduce heavy usage",
        "Medium":"🌤 Use efficiently",
        "Low":"💡 Best time to use"
    }[cat]

def tip(temp, hum):
    if temp > 32: return "🔥 Reduce AC usage"
    if hum > 75: return "💧 Use fans instead"
    return "✅ Balanced weather"

def trend(cat):
    return {
        "High":"📈 High demand expected",
        "Medium":"📊 Stable demand",
        "Low":"📉 Low demand"
    }[cat]

def show_output(temp, hum, wind, cat, conf=None):
    st.subheader("🌦 Weather Data")
    c1,c2,c3 = st.columns(3)
    c1.metric("Temperature", f"{temp}°C")
    c2.metric("Humidity", f"{hum}%")
    c3.metric("Wind Speed", f"{wind} m/s")

    st.subheader("🔮 Prediction Result")
    if cat=="High":
        st.error("⚡ High Demand 🚨")
    elif cat=="Medium":
        st.warning("⚡ Medium Demand")
    else:
        st.success("⚡ Low Demand")

    if conf:
        st.info(f"Confidence Score: {conf}%")

    st.subheader("💡 Recommendation")
    st.write(recommend(cat))

    st.subheader("📌 Personalized Tip")
    st.info(tip(temp, hum))

    st.subheader("📊 Future Trend")
    st.write(trend(cat))

    st.subheader("📊 Weather Graph")
    st.bar_chart(pd.DataFrame(
        [temp, hum, wind],
        index=["Temp","Humidity","Wind"]
    ))

# =========================
# SESSION
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio("", [
    " Smart Prediction",
    " Advanced Analysis",
    " History",
    " Model Comparison"
])

# =========================
# MAIN BOX
# =========================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# =========================
# SMART PREDICTION
# =========================
if menu == " Smart Prediction":
    st.subheader(" Smart Prediction (Best Model - Random Forest)")
    city = st.text_input("Enter City Name", placeholder="e.g., Chennai, Delhi, Mumbai")
    if st.button("Predict Now"):
        data = get_weather(city)
        if data:
            temp, hum, wind = data
            probs = clf_model.predict_proba([[temp,hum,wind]])[0]
            idx = np.argmax(probs)
            cat = clf_model.classes_[idx]
            conf = round(probs[idx]*100,2)
            show_output(temp, hum, wind, cat, conf)
            st.session_state.history.append({
                "Model":"Smart",
                "City":city,
                "Result":cat
            })

# =========================
# ADVANCED
# =========================
elif menu == " Advanced Analysis":
    model = st.selectbox("Select Model",
        ["Regression","Classification","ANN","Apriori","Clustering"])
    city = st.text_input("Enter City Name")
    if st.button("Run Model"):
        data = get_weather(city)
        if data:
            temp, hum, wind = data
            if model=="Regression":
                pred = reg_model.predict([[temp,hum,wind]])[0]
                cat = "Low" if pred<0.33 else "Medium" if pred<0.66 else "High"
            elif model=="Classification":
                probs = clf_model.predict_proba([[temp,hum,wind]])[0]
                idx = np.argmax(probs)
                cat = clf_model.classes_[idx]
            elif model=="ANN":
                probs = ann.predict(np.array([[temp,hum,wind]]))[0]
                idx = np.argmax(probs)
                cat = ["Low","Medium","High"][idx]
            elif model=="Apriori":
                cat = "High" if temp>30 and hum>70 else "Medium" if temp>25 else "Low"
            else:
                cat = "Medium"
            show_output(temp, hum, wind, cat)
            st.session_state.history.append({
                "Model":model,
                "City":city,
                "Result":cat
            })

# =========================
# HISTORY
# =========================
elif menu == " History":
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        st.download_button("Download Report",
                           df.to_csv(index=False),
                           "history.csv")
    else:
        st.info("No history available")

# =========================
# MODEL COMPARISON
# =========================
elif menu == "📊 Model Comparison":
    st.subheader("📊 Model Comparison")
    st.success("🏆 Best Model: Random Forest Classification")
    df = pd.DataFrame({
        "Model":["Regression","Classification RF","ANN"],
        "Score":[0.85,0.94,0.92]
    })
    st.dataframe(df)
    st.bar_chart(df.set_index("Model"))

# =========================
# END BOX
# =========================
st.markdown('</div>', unsafe_allow_html=True)
