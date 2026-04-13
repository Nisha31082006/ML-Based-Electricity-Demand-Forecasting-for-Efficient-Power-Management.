import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =========================
# 🌈 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Electricity Demand System",
    layout="wide"
)

# =========================
# 🎨 FONT + STYLE (WORKING)
# =========================
st.markdown("""
<style>

/* Enter City Name */
div[data-testid="stTextInput"] label {
    font-size: 28px !important;
    font-weight:bold !important;
}

/* Metric Labels */
[data-testid="stMetricLabel"] {
    font-size: 26px !important;
    font-weight: bold !important;
}

/* Metric Values */
[data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: 900 !important;
}

/* Recommendation text */
.reco-text {
    font-size: 28px;
    font-weight: bold;
}

.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🌦 API
# =========================
API_KEY = "9804591c6d0b28e5cd3807041ce86c95"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        st.error("City not found")
        return None

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"]
    }

# =========================
# 🤖 MODEL
# =========================
def train_model():
    df = pd.DataFrame({
        "temp": [20, 30, 35, 25, 15, 40, 28, 32, 22, 18],
        "humidity": [60, 70, 80, 50, 40, 90, 65, 75, 55, 45],
        "wind": [5, 3, 2, 6, 10, 1, 4, 3, 7, 8],
        "load": ["Low", "Medium", "High", "Medium", "Low",
                 "High", "Medium", "High", "Low", "Medium"]
    })

    X = df[["temp", "humidity", "wind"]]
    y = df["load"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# =========================
# 🏠 TITLE
# =========================
st.markdown("""
<h1 style='text-align:center; color:#0D47A1;'>
⚡ Electricity Demand Prediction System
</h1>
""", unsafe_allow_html=True)

# =========================
# 🌍 INPUT
# =========================
city = st.text_input("Enter City Name")

# =========================
# 🔍 BUTTON
# =========================
if st.button("Predict Now"):

    weather = get_weather(city)

    if weather:

        st.success("Weather data fetched")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature", f"{weather['temp']} °C")
        col2.metric("💧 Humidity", f"{weather['humidity']} %")
        col3.metric("🌬 Wind", f"{weather['wind_speed']} m/s")

        # =========================
        #  PREDICTION
        # =========================
        input_data = np.array([[weather["temp"], weather["humidity"], weather["wind_speed"]]])
        prediction = model.predict(input_data)[0]

        st.subheader(" Prediction")

        if prediction == "High":
            st.error("⚡ High Electricity Demand")

            st.markdown("### Recommendation")
            st.markdown(
                "<p class='reco-text' style='color:red;'>⚠ Reduce electricity usage immediately</p>",
                unsafe_allow_html=True
            )

        elif prediction == "Medium":
            st.warning("⚡ Medium Electricity Demand")

            st.markdown("### Recommendation")
            st.markdown(
                "<p class='reco-text' style='color:#f9a825;'>🌤 Use electricity moderately</p>",
                unsafe_allow_html=True
            )

        else:
            st.success("⚡ Low Electricity Demand")

            st.markdown("### Recommendation")
            st.markdown(
                "<p class='reco-text' style='color:green;'>🌟 Good time for heavy usage like washing, charging, etc.</p>",
                unsafe_allow_html=True
            )