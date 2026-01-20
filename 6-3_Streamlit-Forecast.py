"""
🏠 HDB Analytics Suite - Price Forecasting  
-----------------------

Summary:
This Streamlit app provides a comprehensive toolkit for analyzing and forecasting Singapore HDB resale prices.  

 🔮 Price Forecasting  
   - Uses a trained Prophet time-series model to forecast future HDB resale prices for a selected Town and Flat Type.  
   - Produces interactive charts of historical and predicted prices with confidence intervals.  
   - Generates summary tables and key insights (expected trend, price change percentage, forecast values).  

Prerequisites:
- Python 3.9+ (recommended)  
- Install required packages:
    pip install streamlit pandas numpy prophet plotly scikit-learn catboost joblib

- Required data/model files in the same directory:
    • raw_data_main.csv                      → Raw HDB resale dataset  
    • 6-4_prophet_full_pipeline.pkl          → Trained Prophet pipeline for forecasting  

Usage:
    Run the app locally with:
        streamlit run 6-3_Streamlit-Forecast.py


"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import timedelta

# ------------------------------
# Helper Function
# ------------------------------
def format_to_k(price):
    """Formats a numeric price value to a string like '$349K'."""
    if pd.isna(price) or price == 0:
        return "$0K"
    return f"${price/1000:.0f}K"

# ------------------------------
# Load Data & Model
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("raw_data_main.csv")
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH_NUM']].rename(columns={'MONTH_NUM': 'month'}).assign(day=1))
    return df

@st.cache_resource
def load_model():
    with open("6-4_prophet_full_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    
    prophet_model = pipeline['model']
    print("Loaded Prophet Model Parameters:")
    print(" - changepoint_prior_scale:", prophet_model.changepoint_prior_scale)
    print(" - seasonality_prior_scale:", prophet_model.seasonality_prior_scale)
    return pipeline

# Load data and model
df = load_data()
pipeline = load_model()

# Extract components from pipeline
prophet_model = pipeline['model']
town_encoder = pipeline['town_encoder']
flat_type_encoder = pipeline['flat_type_encoder']

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.title("🏙️ HDB Resale Price Forecasting App")
st.sidebar.header("🔧 Forecast Settings")

towns = sorted(df['TOWN'].unique())
flat_types = sorted(df['FLAT_TYPE'].unique())
forecast_options = [6, 12, 18, 24]

selected_town = st.sidebar.selectbox("Select TOWN:", towns)
selected_flat = st.sidebar.selectbox("Select FLAT_TYPE:", flat_types)
forecast_period = st.sidebar.selectbox("Forecast Period (Months):", forecast_options)

# ------------------------------
# Filter Historical Data & Prepare Features
# ------------------------------
df_filtered = df[(df['TOWN'] == selected_town) & (df['FLAT_TYPE'] == selected_flat)].copy()
if df_filtered.empty:
    st.warning("No data found for this TOWN and FLAT_TYPE combination.")
    st.stop()

agg_cols = {
    'RESALE_PRICE': ['mean', 'median', 'count', 'std'],
    'FLOOR_AREA_SQM': 'mean',
    'AGE': 'mean',
    'STOREY_NUMERIC': 'mean',
    'IS_OUTLIERS': 'sum'
}
df_monthly = df_filtered.groupby(['DATE']).agg(agg_cols).round(2)
df_monthly.columns = ['_'.join(col).strip() for col in df_monthly.columns.values]
df_monthly = df_monthly.reset_index()

df_monthly = df_monthly.rename(columns={'DATE': 'ds', 'RESALE_PRICE_mean': 'y'})
df_monthly = df_monthly.sort_values('ds').reset_index(drop=True)

df_monthly['price_volatility'] = df_monthly['y'].rolling(window=3).std().fillna(0)
df_monthly['transaction_volume'] = df_monthly['RESALE_PRICE_count']
df_monthly['avg_floor_area'] = df_monthly['FLOOR_AREA_SQM_mean']
df_monthly['avg_age'] = df_monthly['AGE_mean'] 
df_monthly['avg_storey'] = df_monthly['STOREY_NUMERIC_mean']
df_monthly['outlier_count'] = df_monthly.get('IS_OUTLIERS_sum', 0)
df_monthly['outlier_ratio'] = (df_monthly['outlier_count'] / df_monthly['transaction_volume']).fillna(0)
df_monthly['market_stress'] = df_monthly['price_volatility'] * df_monthly['outlier_ratio']
df_monthly = df_monthly.fillna(0)

# ------------------------------
# Display basic statistics (MODIFIED)
# ------------------------------
st.subheader(f"📈 Historical Data Overview: {selected_town} - {selected_flat}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg Price", format_to_k(df_monthly['y'].mean()))
with col2:
    st.metric("Min Price", format_to_k(df_monthly['y'].min()))
with col3:
    st.metric("Max Price", format_to_k(df_monthly['y'].max()))

# ------------------------------
# Forecast Future Prices
# ------------------------------
local_model = Prophet(
    changepoint_prior_scale=prophet_model.changepoint_prior_scale,
    seasonality_prior_scale=prophet_model.seasonality_prior_scale,
    yearly_seasonality=True,
    daily_seasonality=False,
    weekly_seasonality=False
)
local_model.fit(df_monthly)

future = local_model.make_future_dataframe(periods=forecast_period, freq='MS')
forecast = local_model.predict(future)
forecast_merged = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(df_monthly, on='ds', how='left')

historical_data = forecast_merged[forecast_merged['y'].notna()]
future_data = forecast_merged[forecast_merged['y'].isna()]

# ------------------------------
# Plotting
# ------------------------------
st.subheader(f"📊 Forecasted Resale Prices for {selected_town} - {selected_flat}")

min_year = forecast_merged['ds'].dt.year.min()
max_year = forecast_merged['ds'].dt.year.max()
selected_year_range = st.slider(
    "Select Year Range to Display:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)
start_year, end_year = selected_year_range

chart_hist_data = historical_data[
    (historical_data['ds'].dt.year >= start_year) & (historical_data['ds'].dt.year <= end_year)
]
chart_future_data = future_data[
    (future_data['ds'].dt.year >= start_year) & (future_data['ds'].dt.year <= end_year)
]

fig = go.Figure()
fig.add_trace(go.Scatter(x=chart_hist_data['ds'], y=chart_hist_data['y'], mode='lines+markers', name='Historical Actual', line=dict(color='black', width=2), marker=dict(size=4)))
fig.add_trace(go.Scatter(x=chart_hist_data['ds'], y=chart_hist_data['yhat'], mode='lines', name='Historical Fitted', line=dict(color='blue', width=1, dash='dot'), opacity=0.7))
fig.add_trace(go.Scatter(x=chart_future_data['ds'], y=chart_future_data['yhat'], mode='lines+markers', name='Future Forecast', line=dict(color='red', width=3), marker=dict(size=6)))
fig.add_trace(go.Scatter(x=chart_future_data['ds'], y=chart_future_data['yhat_upper'], mode='lines', name='Confidence Upper', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=chart_future_data['ds'], y=chart_future_data['yhat_lower'], mode='lines', name='Confidence Band', fill='tonexty', line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.2)', showlegend=True))
fig.update_layout(title=f"Resale Price Forecast: {selected_town} - {selected_flat}", xaxis_title="Date", yaxis_title="Resale Price (SGD)", hovermode='x unified', height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Forecast Summary Table (MODIFIED)
# ------------------------------
st.subheader("🔮 Forecast Summary")

if not future_data.empty:
    summary_table = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    summary_table['ds'] = summary_table['ds'].dt.strftime('%Y-%m')
    summary_table.columns = ['Month', 'Forecast', 'Conf_Lower', 'Conf_Upper']
    
    currency_cols = ['Forecast', 'Conf_Lower', 'Conf_Upper']
    for col in currency_cols:
        summary_table[col] = summary_table[col].apply(format_to_k)
    
    st.dataframe(summary_table, use_container_width=True)
    
    # ------------------------------
    # Key insights (MODIFIED)
    # ------------------------------
    st.subheader("💡 Key Insights")
    
    latest_actual = historical_data['y'].iloc[-1] if not historical_data.empty else 0
    last_forecast = future_data['yhat'].iloc[-1] if not future_data.empty else 0
    price_change = last_forecast - latest_actual
    price_change_pct = (price_change / latest_actual * 100) if latest_actual > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", format_to_k(latest_actual), help="Latest available actual price")
    with col2:
        st.metric(
            f"Forecast ({forecast_period}M)", 
            format_to_k(last_forecast),
            delta=f"{price_change_pct:+.1f}%",
            help=f"Forecasted price after {forecast_period} months"
        )
    with col3:
        trend = "📈 Upward" if price_change > 0 else "📉 Downward" if price_change < 0 else "➡️ Stable"
        st.metric("Trend Direction", trend, help="Overall price movement direction")
else:
    st.warning("No future forecasts available.")

# ------------------------------
# Model Information
# ------------------------------
with st.expander("ℹ️ Model Information"):
    st.write(f"""
    **Model Type:** Facebook Prophet
    
    **Parameters:**
    - Changepoint Prior Scale: {prophet_model.changepoint_prior_scale:.5f}
    - Seasonality Prior Scale: {prophet_model.seasonality_prior_scale:.5f}
    - Yearly Seasonality: Enabled
    - Weekly/Daily Seasonality: Disabled
    
    **Forecast Settings:**
    - Selected Location: {selected_town}
    - Selected Flat Type: {selected_flat}
    - Forecast Period: {forecast_period} months
    
    **Data Quality:**
    - Historical Data Points: {len(historical_data)}
    - Date Range: {historical_data['ds'].min().strftime('%Y-%m') if not historical_data.empty else 'N/A'} to {historical_data['ds'].max().strftime('%Y-%m') if not historical_data.empty else 'N/A'}
    """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("*This forecast is for informational purposes only and should not be used as the sole basis for investment decisions.*")