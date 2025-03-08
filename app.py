import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
    .sidebar-content {
        background-color: #1f4e79;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_data(ttl=300)
def load_data():
    file_path = "partc.parquet"  # Ensure the file is in the same directory as main.py
    df = pd.read_parquet(file_path, engine="pyarrow")
    return df

df = load_data()

# Remove companies with less than 180 records
df = df.groupby('names').filter(lambda x: len(x) >= 180)

# Sidebar - Company Selection
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-title">📊 Stock Forecasting</p>', unsafe_allow_html=True)
st.sidebar.header("Select a Company")
companies = df['names'].unique()
selected_company = st.sidebar.selectbox("📌 Choose a company", companies)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Model Selection
st.sidebar.header("Select Model")
model_choice = st.sidebar.radio("📊 Choose a prediction model:", ["Random Forest", "Linear Regression"])

# Filter Data for Selected Company
df_company = df[df['names'] == selected_company]

# Features & Target
features = ["open", "high", "low", "volume", "EMA_10", "MACD", "ATR_14", "Williams_%R"]
target = "close"
X = df_company[features]
y = df_company[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "Linear Regression":
    model = LinearRegression()

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

# Improved Stock Price Forecast Graph with Rolling Average
st.markdown("### 📉 Stock Price Forecast vs Actual Close Prices")
df_company['Rolling_Avg'] = df_company['close'].rolling(window=10).mean()
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['close'], mode='lines', name='Historical Close Prices', line=dict(color='blue')))
forecast_fig.add_trace(go.Scatter(x=df_company['date'], y=df_company['Rolling_Avg'], mode='lines', name='10-day Rolling Avg', line=dict(color='orange', dash='dot')))
forecast_fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Close'], mode='lines+markers', name='Forecasted Close Prices', line=dict(color='red', dash='dash')))
forecast_fig.update_layout(title="Stock Price Forecast vs Actual Close", xaxis_title="Date", yaxis_title="Close Price", template="plotly_dark")
st.plotly_chart(forecast_fig, use_container_width=True)

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
