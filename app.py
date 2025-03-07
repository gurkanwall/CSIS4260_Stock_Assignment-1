import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyarrow.parquet as pq

# Load dataset
@st.cache_data
def load_data():
    file_path = "partc.parquet"  # Ensure the file is in the same directory as main.py
    df = pd.read_parquet(file_path, engine="pyarrow")
    return df

df = load_data()

# Sidebar - Company Selection
st.sidebar.header("Select Company")
companies = df['names'].unique()
selected_company = st.sidebar.selectbox("Choose a company", companies)

df_company = df[df['name'] == selected_company]

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

# Display Metrics
st.metric(label="Mean Absolute Error", value=round(mae, 2))
st.metric(label="Mean Squared Error", value=round(mse, 2))
st.metric(label="R-squared", value=round(r2, 2))

# Forecast Next 20 Days
future_dates = pd.date_range(start=df_company['date'].max(), periods=21, freq='B')[1:]
future_X = X.tail(20)  # Using last known values as reference
future_preds = model.predict(future_X)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Close": np.round(future_preds, 2)
})

# Plotting Results
st.subheader("Stock Price Forecast")
fig, ax = plt.subplots()
ax.plot(df_company['date'], df_company['close'], label="Historical Close Prices")
ax.plot(forecast_df['Date'], forecast_df['Forecasted Close'], linestyle='dashed', marker='o', label="Forecasted Close Prices")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
st.pyplot(fig)

# Display Forecast Table
st.subheader("Forecasted Values for Next 20 Days")
st.dataframe(forecast_df)
