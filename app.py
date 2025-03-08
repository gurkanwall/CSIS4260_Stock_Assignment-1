import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyarrow.parquet as pq

# Set Streamlit Page Configurations
st.set_page_config(page_title="Stock Price Forecasting", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .stTextInput, .stSelectbox { background-color: #ffffff; }
    .metric-container { display: flex; justify-content: space-around; }
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    file_path = "partc.parquet"  # Ensure the file is in the same directory as main.py
    df = pd.read_parquet(file_path, engine="pyarrow")
    return df

df = load_data()

# Sidebar - Company Selection
st.sidebar.title("Stock Forecasting Dashboard")
st.sidebar.header("Select a Company")
companies = df['names'].unique()
selected_company = st.sidebar.selectbox("Choose a company", companies)

# Filter Data for Selected Company
df_company = df[df['names'] == selected_company]

# Features & Target
features = ["open", "high", "low", "volume", "EMA_10", "MACD", "ATR_14", "Williams_%R"]
target = "close"
X = df_company[features]
y = df_company[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Dashboard Layout
st.title("📈 Stock Price Forecasting Dashboard")
st.subheader(f"Company: {selected_company}")

# Display Metrics in a More Engaging Way
st.markdown("### 📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric(label="📉 Mean Absolute Error", value=round(mae, 2))
col2.metric(label="📈 Mean Squared Error", value=round(mse, 2))
col3.metric(label="💡 R-squared Score", value=round(r2, 2))

# Forecast Next 20 Days
future_dates = pd.date_range(start=df_company['date'].max(), periods=21, freq='B')[1:].date  # Remove time part
future_X = X.tail(20)  # Using last known values as reference
future_preds = model.predict(future_X)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Close": np.round(future_preds, 2)
})

# Display Forecast Table at the Top
st.markdown("### 🔮 Forecasted Stock Prices for Next 20 Days")
st.dataframe(forecast_df, hide_index=True, width=800)

# Candlestick Chart
st.markdown("### 📊 Historical Price Data - Candlestick Chart")
candlestick_fig = go.Figure(data=[go.Candlestick(
    x=df_company['date'],
    open=df_company['open'],
    high=df_company['high'],
    low=df_company['low'],
    close=df_company['close'],
    name="Candlestick"
)])
candlestick_fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(candlestick_fig, use_container_width=True)

# Stock Price Forecast Graph
st.markdown("### 📉 Stock Price Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_company['date'], df_company['close'], label="Historical Close Prices", color="blue")
ax.plot(forecast_df['Date'], forecast_df['Forecasted Close'], linestyle='dashed', marker='o', color='red', label="Forecasted Close Prices")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.grid(True)
st.pyplot(fig)

# Interactive Technical Indicators
st.markdown("### 📡 Technical Indicators")
technical_fig = go.Figure()
technical_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['EMA_10'], mode='lines', name='EMA 10', line=dict(color='yellow')))
technical_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['MACD'], mode='lines', name='MACD', line=dict(color='green')))
technical_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['ATR_14'], mode='lines', name='ATR 14', line=dict(color='orange')))
technical_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['Williams_%R'], mode='lines', name='Williams %R', line=dict(color='purple')))
technical_fig.update_layout(title="Technical Indicators Over Time", xaxis_title="Date", yaxis_title="Indicator Value", template="plotly_dark")
st.plotly_chart(technical_fig, use_container_width=True)

# Volume Analysis
st.markdown("### 📊 Volume Analysis")
st.bar_chart(df_company.set_index("date")["volume"])
