"""
🏠 HDB Analytics Suite
-----------------------

Summary:
This Streamlit app provides a comprehensive toolkit for analyzing and forecasting Singapore HDB resale prices.  
It has two main modules:  

1. 💰 Price Estimation  
   - Uses a trained CatBoost machine learning model to estimate the resale price of an HDB flat based on user inputs  
     (Town, Flat Type, Floor Area, Storey Number, and Age).  
   - Provides insights such as price tier (Budget, Mid-range, Premium, Luxury), age group classification, and 
     comparisons against average and median prices for similar flats.  
   - Displays percentile ranking of the predicted price within comparable properties.  

2. 🔮 Price Forecasting  
   - Uses a trained Prophet time-series model to forecast future HDB resale prices for a selected Town and Flat Type.  
   - Produces interactive charts of historical and predicted prices with confidence intervals.  
   - Generates summary tables and key insights (expected trend, price change percentage, forecast values).  

Additional Features:
- Interactive visualizations with Plotly.  
- Sample resale data for quick reference.  
- Clean UI with sidebar navigation.  
- Metrics and insights presented in a dashboard-style layout.  

Prerequisites:
- Python 3.9+ (recommended)  
- Install required packages:
    pip install streamlit pandas numpy prophet plotly scikit-learn catboost joblib

- Required data/model files in the same directory:
    • raw_data_main.csv                      → Raw HDB resale dataset  
    • 4-4_catboost_model_valid.pkl           → Trained CatBoost price prediction model  
    • 4-5_best_catboost_params.json          → Features & parameters for CatBoost model  
    • 6-4_prophet_full_pipeline.pkl          → Trained Prophet pipeline for forecasting  

Usage:
    Run the app locally with:
        streamlit run Streamlit-PriceNForcast.py
    (replace `app.py` with your filename)

"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from prophet import Prophet
import plotly.graph_objs as go
import warnings

warnings.filterwarnings('ignore')

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ====================
# Page Configuration
# ====================
st.set_page_config(
    page_title="HDB Analytics Suite", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# Helper Functions
# ====================
def format_to_k(price):
    """Formats a numeric price value to a string like '$349K'."""
    if pd.isna(price) or price == 0:
        return "$0K"
    return f"${price/1000:.0f}K"

# ====================
# Data Loading Functions
# ====================
@st.cache_data
def load_data():
    df = pd.read_csv("raw_data_main.csv")
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH_NUM']].rename(columns={'MONTH_NUM': 'month'}).assign(day=1))
    return df

@st.cache_resource
def load_forecast_model():
    with open("6-4_prophet_full_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    prophet_model = pipeline['model']
    return pipeline

@st.cache_resource
def load_price_model_and_features():
    model = joblib.load("4-4_catboost_model_valid.pkl")
    with open("4-5_best_catboost_params.json", "r") as f:
        params_data = json.load(f)
    features_used = params_data['features_used']
    return model, features_used

# ====================
# Price Prediction Function
# ====================
def predict_price_model(input_dict, price_model, features_used, df_main):
    input_df = pd.DataFrame([input_dict])
    
    # Compute AGE_GROUP for display
    bins_age = [0, 5, 15, 30, float('inf')]
    labels_age = ['New', 'Moderate', 'Old', 'Very Old']
    age_group = pd.cut(input_df['AGE'], bins=bins_age, labels=labels_age, right=False)[0]
    
    # One-hot encode TOWN and FLAT_TYPE to match training
    for col in features_used:
        if col.startswith('TOWN_'):
            input_df[col] = 1 if col == f"TOWN_{input_dict['TOWN']}" else 0
        elif col.startswith('FLAT_TYPE_'):
            input_df[col] = 1 if col == f"FLAT_TYPE_{input_dict['FLAT_TYPE']}" else 0
        elif col not in input_df.columns:
            input_df[col] = 0

    input_encoded = input_df[features_used]
    price = price_model.predict(input_encoded)[0]
    price = np.expm1(price)

    # Determine PRICE_TIER
    price_bins = df_main['RESALE_PRICE'].quantile([0, 0.25, 0.75, 0.95, 1.0])
    labels_price = ['Budget', 'Mid-range', 'Premium', 'Luxury']
    price_tier = pd.cut([price], bins=price_bins, labels=labels_price, include_lowest=True)[0]

    return price, age_group, price_tier

# ====================
# Main App
# ====================
def main():
    # Load data
    df = load_data()
    
    # Sidebar Navigation
    st.sidebar.title("🏠 HDB Analytics Suite")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["💰 Price Estimation", "🔮 Price Forecasting"]
    )
    
    # ====================
    # PRICE ESTIMATION PAGE
    # ====================
    if app_mode == "💰 Price Estimation":
        st.title("💰 HDB Resale Price Estimation")
        st.markdown("Get instant price estimates for HDB resale flats using machine learning")
        
        # Load price model
        try:
            price_model, features_used = load_price_model_and_features()
        except Exception as e:
            st.error(f"Error loading price model: {e}")
            return
        
        st.subheader("🏠 Enter Property Details")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            est_town = st.selectbox("Select Town", sorted(df["TOWN"].unique()))
            est_flat_type = st.selectbox("Flat Type", df["FLAT_TYPE"].unique())
            est_floor_area = st.slider("Floor Area (sqm)", 60, 160, 90)
        
        with col2:
            est_storey_num = st.slider("Storey Number", 1, 50, 10)
            est_age = st.slider("Flat Age (years)", 0, 60, 15)
        
        # Prediction section
        st.markdown("---")
        
        if st.button("🔍 Estimate Price", type="primary"):
            input_dict = {
                "FLOOR_AREA_SQM": est_floor_area,
                "STOREY_NUMERIC": est_storey_num,
                "AGE": est_age,
                "TOWN": est_town,
                "FLAT_TYPE": est_flat_type
            }
            
            estimated_price, age_group, price_tier = predict_price_model(input_dict, price_model, features_used, df)
            
            # Tier colors
            tier_colors = {
                'Budget': "#27ae60",
                'Mid-range': "#f39c12",
                'Premium': "#e74c3c",
                'Luxury': "#9b59b6"
            }
            
            price_tier_str = str(price_tier) if pd.notna(price_tier) else "Mid-range"
            tier_color = tier_colors.get(price_tier_str, "#95a5a6")
            
            # Display result
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {tier_color}20, {tier_color}10); 
                        border-radius: 20px; border: 3px solid {tier_color}; margin: 20px 0;">
                <h1 style="color: {tier_color}; margin: 0; font-size: 48px;">💰 SGD {estimated_price:,.0f}</h1>
                <h3 style="color: {tier_color}; font-weight: bold; margin: 10px 0;">{price_tier_str} Segment</h3>
                <p style="color: #666; margin: 5px 0; font-size: 16px;">Age Group: {age_group}</p>
                <p style="color: #666; margin: 0; font-size: 16px;">{est_flat_type} in {est_town} • {est_floor_area}sqm • Storey {est_storey_num} • {est_age} years old</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.subheader("📊 Price Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            # Get comparative data for the same town and flat type
            comparable_data = df[(df['TOWN'] == est_town) & (df['FLAT_TYPE'] == est_flat_type)]
            
            if not comparable_data.empty:
                avg_price = comparable_data['RESALE_PRICE'].mean()
                median_price = comparable_data['RESALE_PRICE'].median()
                
                with col1:
                    diff_from_avg = ((estimated_price - avg_price) / avg_price) * 100
                    st.metric(
                        "vs Average", 
                        format_to_k(avg_price),
                        delta=f"{diff_from_avg:+.1f}%"
                    )
                
                with col2:
                    diff_from_median = ((estimated_price - median_price) / median_price) * 100
                    st.metric(
                        "vs Median", 
                        format_to_k(median_price),
                        delta=f"{diff_from_median:+.1f}%"
                    )
                
                with col3:
                    percentile = (comparable_data['RESALE_PRICE'] <= estimated_price).mean() * 100
                    st.metric("Price Percentile", f"{percentile:.0f}%")
        
        # Display sample data for reference
        with st.expander("📋 View Sample Data"):
            sample_data = df.sample(10)[['TOWN', 'FLAT_TYPE', 'FLOOR_AREA_SQM', 'STOREY_NUMERIC', 'AGE', 'RESALE_PRICE']]
            sample_data['RESALE_PRICE'] = sample_data['RESALE_PRICE'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(sample_data, use_container_width=True)
    
    # ====================
    # FORECASTING PAGE
    # ====================
    elif app_mode == "🔮 Price Forecasting":
        st.title("🔮 HDB Resale Price Forecasting")
        st.markdown("Predict future HDB resale prices using advanced time series analysis")
        
        # Load forecast model
        try:
            pipeline = load_forecast_model()
            prophet_model = pipeline['model']
        except Exception as e:
            st.error(f"Error loading forecast model: {e}")
            return
        
        # Forecast Settings
        st.sidebar.header("🔧 Forecast Settings")
        
        towns = sorted(df['TOWN'].unique())
        flat_types = sorted(df['FLAT_TYPE'].unique())
        forecast_options = [6, 12, 18, 24]
        
        selected_town = st.sidebar.selectbox("Select Town:", towns)
        selected_flat = st.sidebar.selectbox("Select Flat Type:", flat_types)
        forecast_period = st.sidebar.selectbox("Forecast Period (Months):", forecast_options)
        
        # Filter and prepare data
        df_filtered = df[(df['TOWN'] == selected_town) & (df['FLAT_TYPE'] == selected_flat)].copy()
        
        if df_filtered.empty:
            st.warning("No data found for this Town and Flat Type combination.")
            return
        
        # Aggregate monthly data
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
        
        # Display overview
        st.subheader(f"📈 Historical Data Overview: {selected_town} - {selected_flat}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Price", format_to_k(df_monthly['y'].mean()))
        with col2:
            st.metric("Min Price", format_to_k(df_monthly['y'].min()))
        with col3:
            st.metric("Max Price", format_to_k(df_monthly['y'].max()))
        with col4:
            st.metric("Data Points", len(df_monthly))
        
        # Generate forecast
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
        
        # Plotting
        st.subheader(f"📊 Forecasted Resale Prices")
        
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
        
        # Forecast Summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔮 Forecast Summary")
            if not future_data.empty:
                summary_table = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                summary_table['ds'] = summary_table['ds'].dt.strftime('%Y-%m')
                summary_table.columns = ['Month', 'Forecast', 'Conf_Lower', 'Conf_Upper']
                
                currency_cols = ['Forecast', 'Conf_Lower', 'Conf_Upper']
                for col in currency_cols:
                    summary_table[col] = summary_table[col].apply(format_to_k)
                
                st.dataframe(summary_table, use_container_width=True)
            else:
                st.warning("No future forecasts available.")
        
        with col2:
            st.subheader("💡 Key Insights")
            if not future_data.empty and not historical_data.empty:
                latest_actual = historical_data['y'].iloc[-1]
                last_forecast = future_data['yhat'].iloc[-1]
                price_change = last_forecast - latest_actual
                price_change_pct = (price_change / latest_actual * 100) if latest_actual > 0 else 0
                
                st.metric("Current Price", format_to_k(latest_actual))
                st.metric(
                    f"Forecast ({forecast_period}M)", 
                    format_to_k(last_forecast),
                    delta=f"{price_change_pct:+.1f}%"
                )
                trend = "📈 Upward" if price_change > 0 else "📉 Downward" if price_change < 0 else "➡️ Stable"
                st.metric("Trend", trend)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*This application is for informational purposes only and should not be used as the sole basis for investment decisions.*"
    )

if __name__ == "__main__":
    main()