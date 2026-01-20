import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
from prophet import Prophet

# ====================
# Enhanced Data Generator
# ====================

@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start="2017-01-01", end="2025-01-01", freq="M")
    
    # Expanded town list for more realistic data
    towns = ['Ang Mo Kio', 'Bedok', 'Bishan', 'Clementi', 'Queenstown', 'Tampines', 
             'Toa Payoh', 'Woodlands', 'Jurong East', 'Hougang', 'Yishun', 'Pasir Ris']
    flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
    storey_ranges = ['01-05', '06-10', '11-15', '16-20', '21-25', '26-30']
    flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Premium Apartment']
    
    data = {
        'MONTH': np.random.choice(dates, n_samples),
        'TOWN': np.random.choice(towns, n_samples),
        'FLAT_TYPE': np.random.choice(flat_types, n_samples),
        'STOREY_RANGE': np.random.choice(storey_ranges, n_samples),
        'FLOOR_AREA_SQM': np.random.normal(90, 25, n_samples).astype(int),
        'FLAT_MODEL': np.random.choice(flat_models, n_samples),
        'LEASE_COMMENCE_DATE': np.random.randint(1980, 2020, n_samples),
        'RESALE_PRICE': np.random.randint(200000, 900000, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['AGE'] = 2024 - df['LEASE_COMMENCE_DATE']
    
    # Create realistic price relationships
    # Premium towns
    df.loc[df['TOWN'].isin(['Queenstown', 'Bishan', 'Clementi']), 'RESALE_PRICE'] *= 1.4
    # Budget towns
    df.loc[df['TOWN'].isin(['Woodlands', 'Yishun', 'Jurong East']), 'RESALE_PRICE'] *= 0.9
    # Flat type multipliers
    df.loc[df['FLAT_TYPE'] == 'EXECUTIVE', 'RESALE_PRICE'] *= 1.3
    df.loc[df['FLAT_TYPE'] == '3 ROOM', 'RESALE_PRICE'] *= 0.8
    df.loc[df['FLAT_TYPE'] == '5 ROOM', 'RESALE_PRICE'] *= 1.15
    # Higher floors premium
    df.loc[df['STOREY_RANGE'].isin(['21-25', '26-30']), 'RESALE_PRICE'] *= 1.1
    # Newer flats premium
    df.loc[df['AGE'] < 20, 'RESALE_PRICE'] *= 1.2
    # Size factor
    df['RESALE_PRICE'] = df['RESALE_PRICE'] + (df['FLOOR_AREA_SQM'] - 90) * 3000
    
    # Ensure reasonable floor areas
    df.loc[df['FLAT_TYPE'] == '3 ROOM', 'FLOOR_AREA_SQM'] = np.clip(df.loc[df['FLAT_TYPE'] == '3 ROOM', 'FLOOR_AREA_SQM'], 60, 90)
    df.loc[df['FLAT_TYPE'] == '4 ROOM', 'FLOOR_AREA_SQM'] = np.clip(df.loc[df['FLAT_TYPE'] == '4 ROOM', 'FLOOR_AREA_SQM'], 80, 110)
    df.loc[df['FLAT_TYPE'] == '5 ROOM', 'FLOOR_AREA_SQM'] = np.clip(df.loc[df['FLAT_TYPE'] == '5 ROOM', 'FLOOR_AREA_SQM'], 100, 140)
    df.loc[df['FLAT_TYPE'] == 'EXECUTIVE', 'FLOOR_AREA_SQM'] = np.clip(df.loc[df['FLAT_TYPE'] == 'EXECUTIVE', 'FLOOR_AREA_SQM'], 120, 160)
    
    # Create affordability tiers
    df['AFFORDABILITY_TIER'] = 'Mid-Range'
    df.loc[df['RESALE_PRICE'] < 400000, 'AFFORDABILITY_TIER'] = 'Budget-Friendly'
    df.loc[df['RESALE_PRICE'] >= 550000, 'AFFORDABILITY_TIER'] = 'Premium'
    df.loc[df['RESALE_PRICE'] >= 700000, 'AFFORDABILITY_TIER'] = 'Luxury'
    
    return df

# ====================
# Price Estimator (Enhanced)
# ====================

def predict_price(town, flat_type, floor_area, storey, age):
    base_price = 300000
    
    # Town multipliers
    town_multipliers = {
        'Queenstown': 1.5, 'Bishan': 1.4, 'Clementi': 1.3, 'Toa Payoh': 1.2,
        'Ang Mo Kio': 1.1, 'Bedok': 1.1, 'Tampines': 1.0, 'Hougang': 1.0,
        'Pasir Ris': 0.95, 'Jurong East': 0.9, 'Yishun': 0.85, 'Woodlands': 0.8
    }
    
    # Flat type multipliers
    type_multipliers = {
        '3 ROOM': 0.8, '4 ROOM': 1.0, '5 ROOM': 1.2, 'EXECUTIVE': 1.4
    }
    
    price = base_price * town_multipliers.get(town, 1.0) * type_multipliers.get(flat_type, 1.0)
    price += (floor_area - 90) * 3000
    price -= age * 2000
    
    # Storey premium
    if any(high_floor in storey for high_floor in ['16-20', '21-25', '26-30']):
        price += 25000
    elif any(mid_floor in storey for mid_floor in ['11-15']):
        price += 15000
    elif any(low_floor in storey for low_floor in ['06-10']):
        price += 10000
    
    return max(price, 150000)

# ====================
# Forecasting
# ====================

def forecast_prices(df, town, flat_type, horizon_months):
    subset = df[(df["TOWN"] == town) & (df["FLAT_TYPE"] == flat_type)]
    if subset.empty:
        return pd.DataFrame(columns=["ds","y"]), pd.DataFrame()
    
    # Get monthly averages and ensure we have enough data
    avg_prices = subset.groupby("MONTH")["RESALE_PRICE"].mean().reset_index()
    avg_prices = avg_prices.rename(columns={"MONTH": "ds", "RESALE_PRICE": "y"})
    avg_prices = avg_prices.sort_values('ds').reset_index(drop=True)
    
    # Check if we have sufficient data
    if len(avg_prices) < 12:
        st.warning(f"⚠️ Limited data ({len(avg_prices)} months) may result in unreliable forecasts. Prophet works best with 12+ months of data.")
    
    # Create more stable Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        interval_width=0.8,  # Narrower confidence intervals
        changepoint_prior_scale=0.1,  # Reduce trend flexibility
        seasonality_prior_scale=0.1,   # Reduce seasonality flexibility
        n_changepoints=min(5, len(avg_prices)//4)  # Limit changepoints based on data length
    )
    
    # Add monthly seasonality if we have enough data
    if len(avg_prices) >= 24:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Fit model with error handling
    try:
        model.fit(avg_prices)
        future = model.make_future_dataframe(periods=horizon_months, freq="M")
        forecast = model.predict(future)
        
        # Smooth out extreme variations in forecast
        forecast_start_idx = len(avg_prices)
        if forecast_start_idx < len(forecast):
            # Apply smoothing to reduce dramatic swings
            forecast_values = forecast['yhat'][forecast_start_idx:].values
            if len(forecast_values) > 1:
                # Apply simple moving average smoothing
                smoothed = np.convolve(forecast_values, np.ones(min(3, len(forecast_values)))/min(3, len(forecast_values)), mode='same')
                forecast.loc[forecast_start_idx:, 'yhat'] = smoothed
                
                # Adjust confidence intervals proportionally
                forecast.loc[forecast_start_idx:, 'yhat_lower'] = forecast.loc[forecast_start_idx:, 'yhat'] * 0.85
                forecast.loc[forecast_start_idx:, 'yhat_upper'] = forecast.loc[forecast_start_idx:, 'yhat'] * 1.15
        
        return avg_prices, forecast
        
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return avg_prices, pd.DataFrame()

# ====================
# Town Coordinates
# ====================

town_coords = {
    "Ang Mo Kio": {"lat": 1.3691, "lon": 103.8454},
    "Bedok": {"lat": 1.3244, "lon": 103.9302},
    "Bishan": {"lat": 1.3521, "lon": 103.8498},
    "Clementi": {"lat": 1.3162, "lon": 103.7649},
    "Queenstown": {"lat": 1.2942, "lon": 103.7861},
    "Tampines": {"lat": 1.3496, "lon": 103.9568},
    "Toa Payoh": {"lat": 1.3343, "lon": 103.8474},
    "Woodlands": {"lat": 1.4382, "lon": 103.7890},
    "Jurong East": {"lat": 1.3329, "lon": 103.7436},
    "Hougang": {"lat": 1.3612, "lon": 103.8863},
    "Yishun": {"lat": 1.4304, "lon": 103.8354},
    "Pasir Ris": {"lat": 1.3721, "lon": 103.9474}
}

# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="HDB Insights Pro", page_icon="🏘️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #1f4e79; margin-bottom: 10px; }
    .sub-header { font-size: 20px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #3498db; }
    .tier-budget { border-left-color: #27ae60 !important; }
    .tier-mid { border-left-color: #f39c12 !important; }
    .tier-premium { border-left-color: #e74c3c !important; }
    .tier-luxury { border-left-color: #9b59b6 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏘️ Singapore HDB Insights Pro</div>', unsafe_allow_html=True)
st.markdown("**Comprehensive HDB market analysis with price prediction, market segments, and forecasting**")

# Load data
df = load_data()

# Main tabs
tabs = st.tabs(["🏠 Price Estimator", "📊 Market Insights", "📈 Forecast Analysis"])

# ==================== 
# Tab 1: Price Estimator
# ====================

with tabs[0]:
    st.markdown('<div class="sub-header">🏠 HDB Price Estimator</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        est_town = st.selectbox("Select Town", sorted(df["TOWN"].unique()), key="est_town")
        est_flat_type = st.selectbox("Flat Type", df["FLAT_TYPE"].unique(), key="est_flat_type")
        est_floor_area = st.slider("Floor Area (sqm)", 60, 160, 90, key="est_area")
        
    with col2:
        st.subheader("Additional Features")
        est_storey = st.selectbox("Storey Range", df["STOREY_RANGE"].unique(), key="est_storey")
        est_age = st.slider("Flat Age (years)", 0, 50, 15, key="est_age")
        
        if st.button("🔍 Estimate Price", key="estimate_btn", type="primary"):
            estimated_price = predict_price(est_town, est_flat_type, est_floor_area, est_storey, est_age)
            
            # Determine tier
            if estimated_price < 400000:
                tier = "Budget-Friendly"
                tier_color = "#27ae60"
            elif estimated_price < 550000:
                tier = "Mid-Range"
                tier_color = "#f39c12"
            elif estimated_price < 700000:
                tier = "Premium"
                tier_color = "#e74c3c"
            else:
                tier = "Luxury"
                tier_color = "#9b59b6"
    
    if 'estimated_price' in locals():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {tier_color}20, {tier_color}10); 
                        border-radius: 15px; border: 2px solid {tier_color};">
                <h2 style="color: {tier_color}; margin: 0;">💰 SGD {estimated_price:,.0f}</h2>
                <p style="color: {tier_color}; font-weight: bold; margin: 5px 0;">{tier} Segment</p>
                <p style="color: #666; margin: 0; font-size: 14px;">{est_flat_type} in {est_town} • {est_floor_area}sqm • {est_age} years old</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== 
# Tab 2: Market Insights
# ====================

with tabs[1]:
    st.markdown('<div class="sub-header">📊 Market Segments Analysis</div>', unsafe_allow_html=True)
    
    # Market Insights specific filters
    st.subheader("🔍 Analysis Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        selected_towns = st.multiselect("Select Towns", 
                                       options=sorted(df['TOWN'].unique()), 
                                       default=sorted(df['TOWN'].unique())[0:4],
                                       key="market_towns")
    with filter_col2:
        selected_flat_types = st.multiselect("Select Flat Types", 
                                            options=df['FLAT_TYPE'].unique(), 
                                            default=df['FLAT_TYPE'].unique(),
                                            key="market_types")
    with filter_col3:
        price_range = st.slider("Price Range (SGD)", 
                               min_value=int(df['RESALE_PRICE'].min()), 
                               max_value=int(df['RESALE_PRICE'].max()), 
                               value=(250000, 700000),
                               key="market_price")
    
    # Filter data based on selections
    filtered_df = df[(df['TOWN'].isin(selected_towns)) & 
                     (df['FLAT_TYPE'].isin(selected_flat_types)) &
                     (df['RESALE_PRICE'] >= price_range[0]) & 
                     (df['RESALE_PRICE'] <= price_range[1])]
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for selected filters. Please adjust your selection.")
    else:
        # Market segments overview
        st.subheader("🏘️ Market Segments Overview")
        
        tier_stats = filtered_df.groupby('AFFORDABILITY_TIER').agg({
            'RESALE_PRICE': ['mean', 'count'],
            'FLOOR_AREA_SQM': 'mean',
            'AGE': 'mean'
        }).round(1)
        
        tier_stats.columns = ['avg_price', 'count', 'avg_size', 'avg_age']
        tier_stats = tier_stats.reset_index()
        
        # Display metrics in cards
        cols = st.columns(4)
        tier_colors = ['#27ae60', '#f39c12', '#e74c3c', '#9b59b6']
        tier_names = ['Budget-Friendly', 'Mid-Range', 'Premium', 'Luxury']
        
        for i, (col, tier, color) in enumerate(zip(cols, tier_names, tier_colors)):
            tier_data = tier_stats[tier_stats['AFFORDABILITY_TIER'] == tier]
            
            with col:
                if not tier_data.empty:
                    avg_price = tier_data['avg_price'].iloc[0]
                    count = int(tier_data['count'].iloc[0])
                    avg_size = tier_data['avg_size'].iloc[0]
                    avg_age = tier_data['avg_age'].iloc[0]
                    
                    st.markdown(f"""
                    <div class="metric-card tier-{tier.lower().replace('-', '')}">
                        <h4 style="color: {color}; margin: 0 0 10px 0;">{tier}</h4>
                        <h3 style="margin: 0; color: #2c3e50;">SGD {avg_price:,.0f}</h3>
                        <p style="margin: 5px 0; color: #666; font-size: 12px;">
                            {count} units • {avg_size:.0f}sqm avg • {avg_age:.0f}yr avg
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #999; margin: 0 0 10px 0;">{tier}</h4>
                        <p style="color: #999;">No data</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive charts
        chart_tabs = st.tabs(["💰 Price vs Size", "🗺️ Geographic Distribution", "📈 Market Comparison"])
        
        with chart_tabs[0]:
            st.subheader("Price vs Floor Area Analysis")
            fig = px.scatter(filtered_df, x='FLOOR_AREA_SQM', y='RESALE_PRICE', 
                           color='AFFORDABILITY_TIER',
                           hover_data=['TOWN', 'FLAT_TYPE', 'AGE'],
                           labels={'FLOOR_AREA_SQM': 'Floor Area (sqm)', 
                                  'RESALE_PRICE': 'Resale Price (SGD)',
                                  'AFFORDABILITY_TIER': 'Market Segment'},
                           color_discrete_map={
                               'Budget-Friendly': '#27ae60',
                               'Mid-Range': '#f39c12', 
                               'Premium': '#e74c3c',
                               'Luxury': '#9b59b6'
                           })
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[1]:
            st.subheader("Market Segments by Town")
            town_segment = filtered_df.groupby(['TOWN', 'AFFORDABILITY_TIER']).size().reset_index(name='COUNT')
            fig = px.bar(town_segment, x='TOWN', y='COUNT', color='AFFORDABILITY_TIER',
                        barmode='stack',
                        labels={'TOWN': 'Town', 'COUNT': 'Number of Units'},
                        color_discrete_map={
                            'Budget-Friendly': '#27ae60',
                            'Mid-Range': '#f39c12', 
                            'Premium': '#e74c3c',
                            'Luxury': '#9b59b6'
                        })
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[2]:
            st.subheader("Price Distribution by Flat Type")
            fig = px.box(filtered_df, x='AFFORDABILITY_TIER', y='RESALE_PRICE', 
                        color='FLAT_TYPE',
                        labels={'AFFORDABILITY_TIER': 'Market Segment', 
                               'RESALE_PRICE': 'Resale Price (SGD)'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
 
# ==================== 
# Tab 3: Forecast Analysis
# ====================

with tabs[2]:
    st.markdown('<div class="sub-header">📈 Price Forecast Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_town = st.selectbox("Select Town", sorted(df["TOWN"].unique()), key="forecast_town")
        forecast_flat_type = st.selectbox("Select Flat Type", df["FLAT_TYPE"].unique(), key="forecast_type")
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (months)", 6, 36, 12, key="forecast_horizon")
        
    if st.button("📊 Generate Forecast", key="forecast_btn", type="primary"):
        with st.spinner("Generating forecast..."):
            hist, forecast = forecast_prices(df, forecast_town, forecast_flat_type, forecast_horizon)
            
            if hist.empty:
                st.warning(f"⚠️ No historical data found for {forecast_town} ({forecast_flat_type})")
            else:
                # Forecast visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(hist["ds"], hist["y"], label="Historical", color="darkblue", linewidth=2, marker='o')
                ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="red", linewidth=2)
                ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], 
                              alpha=0.3, color="lightcoral", label="Confidence Interval")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (SGD)")
                ax.set_title(f"Price Forecast: {forecast_town} ({forecast_flat_type})")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Forecast summary metrics
                st.subheader("📊 Forecast Summary")
                latest_price = hist["y"].iloc[-1]
                forecast_price = forecast["yhat"].iloc[-1]
                price_change = ((forecast_price - latest_price) / latest_price) * 100
                confidence_lower = forecast["yhat_lower"].iloc[-1]
                confidence_upper = forecast["yhat_upper"].iloc[-1]
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Current Price", f"SGD {latest_price:,.0f}")
                with metric_cols[1]:
                    st.metric("Forecasted Price", f"SGD {forecast_price:,.0f}", f"{price_change:+.1f}%")
                with metric_cols[2]:
                    st.metric("Lower Bound", f"SGD {confidence_lower:,.0f}")
                with metric_cols[3]:
                    st.metric("Upper Bound", f"SGD {confidence_upper:,.0f}")
                
                # Download option
                st.subheader("📥 Export Data")
                csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
                st.download_button(
                    "⬇️ Download Forecast Data", 
                    csv.encode('utf-8'),
                    f"forecast_{forecast_town}_{forecast_flat_type}.csv",
                    "text/csv"
                )

# Footer
st.markdown("---")
st.markdown("💡 **Insights Guide**: Use Price Estimator for individual valuations, Market Insights for segment analysis, and Forecast for investment planning.")
st.caption("Note: This application uses simulated data for demonstration purposes.")