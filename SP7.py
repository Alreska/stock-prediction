import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import tensorflow as tf
import joblib 

# ----------------------------
# Parameter dataset
# ----------------------------
START = "2016-01-01"
#TODAY = "2025-09-12"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 Stock Prediction")

#stocks = ("UNTR.JK")
stocks = ("UNTR.JK", "ADRO.JK")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# ----------------------------
# Load data dari RapidAPI
# ----------------------------
@st.cache_data
def load_data_rapidapi(ticker, interval="1d", range_="1y"):
    url = "https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart"
    querystring = {"region":"US", "range": range_, "symbol": ticker, "interval": interval}
    headers = {
        "x-rapidapi-key": "727630e79cmsh457ac61b747d6bfp144dfejsn0d7c69f3883f",   
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    if "chart" not in data or not data["chart"]["result"]:
        return pd.DataFrame()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    indicators = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s"),
        "Open": indicators["open"],
        "High": indicators["high"],
        "Low": indicators["low"],
        "Close": indicators["close"],
        "Volume": indicators["volume"]
    })
    return df.dropna()

data_load_state = st.text("Load data...")
data = load_data_rapidapi(selected_stock, interval="1d", range_="1y")
data_load_state.text("Loading data...done ✅")

if data.empty:
    st.error("⚠️ Data kosong. Coba ganti ticker lain atau periksa API Key.")
    st.stop()

# ----------------------------
# Plot data historis
# ----------------------------
st.subheader("Grafik Historis Harga Close")

fig = px.line(data, x="Date", y="Close",
              title=f"Historis Harga Penutupan {selected_stock}")
fig.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Harga",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader('Data saham')
st.write(data.tail())

# ----------------------------
# Preprocessing dengan scaler hasil training
# ----------------------------
scaler_X = joblib.load("scaler_X.pkl")   # scaler untuk input (fit di Colab)
scaler_y = joblib.load("scaler_y.pkl")   # scaler untuk output/target

close_data = data[['Close']].values

# transform pakai scaler training
scaled_data = scaler_X.transform(close_data)

def create_dataset(series, n_steps=4):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps, 0])
        y.append(series[i+n_steps, 0])
    return np.array(X), np.array(y)

n_steps = 4
X, y = create_dataset(scaled_data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ----------------------------
# Load Model GRU hasil training (PSO)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_gpso.keras")

model = load_model()

# ----------------------------
# Prediksi hari besok
# ----------------------------
last_days = scaled_data[-n_steps:]
last_days = last_days.reshape((1, n_steps, 1))

pred_scaled = model.predict(last_days)
pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]   # ✅ pakai scaler_y

st.subheader(f"📌 Prediksi harga {selected_stock} untuk BESOK:")
st.success(f"{pred_price:.2f}")

