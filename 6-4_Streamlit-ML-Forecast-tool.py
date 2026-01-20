"""
📊 HDB Resale Price Model Comparison & Forecasting Tool
-------------------------------------------------------

Purpose:
This Streamlit app helps users analyze HDB resale price trends for a specific town and flat type, compare three Prophet-based forecasting models, and select the best model based on validation performance.

Key Features:
1. Data Selection:
   - Users can select a TOWN and FLAT_TYPE from the sidebar.
   - Data is filtered and aggregated monthly with additional features (e.g., price volatility, transaction volume, average age).

2. Model Comparison:
   - Three Prophet models are trained on the historical data:
     Option 1: Simple Model (time-based only)
     Option 2: High Flexibility (time + regressors: volatility, volume, age)
     Option 3: Key Regressors (time + regressors: age, volume)
   - Models are evaluated on last 20% of data for validation using MAE, RMSE, and MAPE.
   - Best model (lowest RMSE) is highlighted.

3. Model Explanation:
   - Provides a clear description of each model’s strengths, weaknesses, and recommended use case.

4. Future Forecasting:
   - Forecasts the next 12 months using all three models.
   - Plots recent historical data and predicted future prices.
   - Confidence bands are shown for the selected best model.

5. Recommendations & Guidance:
   - Suggests the best model based on validation metrics.
   - Provides performance quality interpretation (Excellent, Good, Fair, Poor).

6. Model Saving:
   - Allows users to save the recommended model as a pickle file.
   - Saved package includes trained model, regressors used, hyperparameters, performance metrics, last known values, and metadata. """

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.title("📊 Model Comparison & Selection Guide")

# Load data (same as before)
@st.cache_data
def load_data():
    df = pd.read_csv("raw_data_main.csv")
    df['DATE'] = pd.to_datetime(
        df[['YEAR', 'MONTH_NUM']].rename(columns={'MONTH_NUM': 'month'}).assign(day=1)
    )
    return df

df = load_data()

# Sidebar for selection
st.sidebar.header("🔧 Select Data")
towns = sorted(df['TOWN'].unique())
flat_types = sorted(df['FLAT_TYPE'].unique())

selected_town = st.sidebar.selectbox("Select TOWN:", towns, index=0)
selected_flat = st.sidebar.selectbox("Select FLAT_TYPE:", flat_types, index=0)

# Filter and prepare data
df_filtered = df[(df['TOWN'] == selected_town) & (df['FLAT_TYPE'] == selected_flat)].copy()

if df_filtered.empty:
    st.warning("No data found for this combination.")
    st.stop()

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

# Add features
df_monthly['price_volatility'] = df_monthly['y'].rolling(window=3, min_periods=1).std().fillna(0)
df_monthly['transaction_volume'] = df_monthly['RESALE_PRICE_count']
df_monthly['avg_floor_area'] = df_monthly['FLOOR_AREA_SQM_mean']
df_monthly['avg_age'] = df_monthly['AGE_mean']
df_monthly['avg_storey'] = df_monthly['STOREY_NUMERIC_mean']
df_monthly['outlier_count'] = df_monthly.get('IS_OUTLIERS_sum', 0)
df_monthly['outlier_ratio'] = (df_monthly['outlier_count'] / df_monthly['transaction_volume']).fillna(0)
df_monthly['market_stress'] = df_monthly['price_volatility'] * df_monthly['outlier_ratio']
df_monthly['price_ma_3'] = df_monthly['y'].rolling(window=3, min_periods=1).mean()
df_monthly['price_ma_6'] = df_monthly['y'].rolling(window=6, min_periods=1).mean()
df_monthly = df_monthly.fillna(method='ffill').fillna(method='bfill').fillna(0)

st.header(f"🏘️ Analysis for {selected_town} - {selected_flat}")
st.write(f"**Data points**: {len(df_monthly)} | **Date range**: {df_monthly['ds'].min().date()} to {df_monthly['ds'].max().date()}")

# Create all three models and compare
@st.cache_resource
def create_and_compare_models(town, flat_type):
    """Create all three model options and compare their performance"""
    
    # Use last 80% for training, last 20% for validation
    train_size = int(len(df_monthly) * 0.8)
    train_data = df_monthly.iloc[:train_size].copy()
    val_data = df_monthly.iloc[train_size:].copy()
    
    models = {}
    forecasts = {}
    performance = {}
    
    # Option 1: Simple Model (No Regressors)
    model1 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        interval_width=0.8
    )
    
    model1.fit(train_data[['ds', 'y']])
    future1 = model1.make_future_dataframe(periods=len(val_data), freq='MS')
    forecast1 = model1.predict(future1)
    models['Option 1: Simple (No Regressors)'] = model1
    forecasts['Option 1: Simple (No Regressors)'] = forecast1
    
    # Calculate performance on validation data
    val_pred1 = forecast1.iloc[train_size:]['yhat'].values
    val_actual = val_data['y'].values
    performance['Option 1: Simple (No Regressors)'] = {
        'MAE': mean_absolute_error(val_actual, val_pred1),
        'RMSE': np.sqrt(mean_squared_error(val_actual, val_pred1)),
        'MAPE': np.mean(np.abs((val_actual - val_pred1) / val_actual)) * 100
    }
    
    # Option 2: High Flexibility
    model2 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,  # Much more flexible
        seasonality_prior_scale=10.0,
        interval_width=0.8
    )
    
    regressors2 = ['price_volatility', 'transaction_volume', 'avg_age']
    for reg in regressors2:
        if reg in train_data.columns:
            model2.add_regressor(reg)
    
    train_data2 = train_data[['ds', 'y'] + [reg for reg in regressors2 if reg in train_data.columns]].copy()
    model2.fit(train_data2)
    
    future2 = model2.make_future_dataframe(periods=len(val_data), freq='MS')
    for reg in regressors2:
        if reg in df_monthly.columns:
            # Use actual values for historical, extrapolate for future
            for i in range(len(future2)):
                if i < len(train_data):
                    future2.loc[i, reg] = train_data.iloc[i][reg]
                else:
                    if reg == 'avg_age':
                        months_ahead = i - len(train_data) + 1
                        future2.loc[i, reg] = train_data[reg].iloc[-1] + (months_ahead / 12)
                    else:
                        future2.loc[i, reg] = train_data[reg].iloc[-1]
    
    forecast2 = model2.predict(future2)
    models['Option 2: High Flexibility'] = model2
    forecasts['Option 2: High Flexibility'] = forecast2
    
    val_pred2 = forecast2.iloc[train_size:]['yhat'].values
    performance['Option 2: High Flexibility'] = {
        'MAE': mean_absolute_error(val_actual, val_pred2),
        'RMSE': np.sqrt(mean_squared_error(val_actual, val_pred2)),
        'MAPE': np.mean(np.abs((val_actual - val_pred2) / val_actual)) * 100
    }
    
    # Option 3: Key Regressors Only
    model3 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=5.0,
        interval_width=0.8
    )
    
    regressors3 = ['avg_age', 'transaction_volume']
    for reg in regressors3:
        if reg in train_data.columns:
            model3.add_regressor(reg)
    
    train_data3 = train_data[['ds', 'y'] + [reg for reg in regressors3 if reg in train_data.columns]].copy()
    model3.fit(train_data3)
    
    future3 = model3.make_future_dataframe(periods=len(val_data), freq='MS')
    for reg in regressors3:
        if reg in df_monthly.columns:
            for i in range(len(future3)):
                if i < len(train_data):
                    future3.loc[i, reg] = train_data.iloc[i][reg]
                else:
                    if reg == 'avg_age':
                        months_ahead = i - len(train_data) + 1
                        future3.loc[i, reg] = train_data[reg].iloc[-1] + (months_ahead / 12)
                    else:
                        future3.loc[i, reg] = train_data[reg].iloc[-1]
    
    forecast3 = model3.predict(future3)
    models['Option 3: Key Regressors'] = model3
    forecasts['Option 3: Key Regressors'] = forecast3
    
    val_pred3 = forecast3.iloc[train_size:]['yhat'].values
    performance['Option 3: Key Regressors'] = {
        'MAE': mean_absolute_error(val_actual, val_pred3),
        'RMSE': np.sqrt(mean_squared_error(val_actual, val_pred3)),
        'MAPE': np.mean(np.abs((val_actual - val_pred3) / val_actual)) * 100
    }
    
    return models, forecasts, performance, train_size

# Create models and get results
models, forecasts, performance, train_size = create_and_compare_models(selected_town, selected_flat)

# Performance comparison table
st.subheader("📊 Model Performance Comparison")

performance_df = pd.DataFrame(performance).T
performance_df = performance_df.round(2)

# Add interpretation
performance_df['MAE_fmt'] = performance_df['MAE'].apply(lambda x: f"${x:,.0f}")
performance_df['RMSE_fmt'] = performance_df['RMSE'].apply(lambda x: f"${x:,.0f}")
performance_df['MAPE_fmt'] = performance_df['MAPE'].apply(lambda x: f"{x:.1f}%")

display_df = performance_df[['MAE_fmt', 'RMSE_fmt', 'MAPE_fmt']].copy()
display_df.columns = ['Mean Abs Error', 'Root Mean Sq Error', 'Mean Abs % Error']

st.dataframe(display_df)

# Find best model
best_model_name = performance_df['RMSE'].idxmin()
st.success(f"🏆 **Best Model (Lowest RMSE)**: {best_model_name}")

# Explanation of each model
st.subheader("🤔 How Each Model Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Option 1: Simple Model**
    - ✅ No regressors (only time-based patterns)
    - ✅ Less prone to overfitting
    - ✅ Robust and reliable
    - ❌ May miss external factor impacts
    - **Best for**: Stable trends, long-term forecasts
    """)

with col2:
    st.markdown("""
    **Option 2: High Flexibility**
    - ✅ Captures complex patterns and trends
    - ✅ Uses key regressors (volatility, volume, age)
    - ✅ More responsive to recent changes
    - ❌ May overfit to recent data
    - **Best for**: Dynamic markets, short-term forecasts
    """)

with col3:
    st.markdown("""
    **Option 3: Key Regressors**
    - ✅ Balance between simplicity and complexity
    - ✅ Uses most important factors (age, volume)
    - ✅ Moderate flexibility
    - ❌ May miss some patterns
    - **Best for**: General purpose, medium-term forecasts
    """)

'''
# Visual comparison
st.subheader("📈 Visual Comparison")

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=df_monthly['ds'], y=df_monthly['y'],
    mode='lines', name='Historical Actual',
    line=dict(color='black', width=2)
))

# Training/validation split line
split_date = df_monthly.iloc[train_size]['ds'].to_pydatetime()  # converts pandas Timestamp to datetime
fig.add_vline(x=split_date, line_dash="dash", line_color="gray", 
              annotation_text="Train/Val Split")

colors = ['red', 'blue', 'green']
for i, (name, forecast) in enumerate(forecasts.items()):
    # Show only validation period forecasts for clarity
    val_forecast = forecast.iloc[train_size:]
    fig.add_trace(go.Scatter(
        x=val_forecast['ds'], y=val_forecast['yhat'],
        mode='lines+markers', name=name,
        line=dict(color=colors[i], width=2),
        marker=dict(size=4)
    ))

fig.update_layout(
    title="Model Comparison on Validation Data",
    xaxis_title="Date", yaxis_title="Resale Price (SGD)",
    height=600, hovermode='x unified',
    yaxis=dict(tickformat='$,.0f')
)

st.plotly_chart(fig, use_container_width=True)
'''
# Future forecast comparison
st.subheader("🔮 Future Forecast Comparison (Next 12 Months)")

colors = ['red', 'blue', 'green']  # Add this here since Visual Comparison was removed
future_forecasts = {}
for i, (name, model) in enumerate(models.items()):
    future = model.make_future_dataframe(periods=12, freq='MS')
    
    # Fill regressors based on model type
    if 'Simple' not in name:
        if 'High Flexibility' in name:
            regressors = ['price_volatility', 'transaction_volume', 'avg_age']
        else:  # Key Regressors
            regressors = ['avg_age', 'transaction_volume']
        
        for reg in regressors:
            if reg in df_monthly.columns:
                last_val = df_monthly[reg].iloc[-1]
                if reg == 'avg_age':
                    for j in range(len(future)):
                        if future.loc[j, 'ds'] > df_monthly['ds'].max():
                            months_ahead = (future.loc[j, 'ds'] - df_monthly['ds'].max()).days // 30
                            future.loc[j, reg] = last_val + (months_ahead / 12)
                        else:
                            matching_row = df_monthly[df_monthly['ds'] == future.loc[j, 'ds']]
                            future.loc[j, reg] = matching_row[reg].iloc[0] if not matching_row.empty else last_val
                else:
                    future[reg] = last_val
    
    forecast_future = model.predict(future)
    future_forecasts[name] = forecast_future.tail(12)

# Plot future forecasts
fig_future = go.Figure()

# Recent historical (last 24 months)
recent_hist = df_monthly.tail(24)
fig_future.add_trace(go.Scatter(
    x=recent_hist['ds'], y=recent_hist['y'],
    mode='lines', name='Recent Historical',
    line=dict(color='black', width=2)
))

# Future forecasts
for i, (name, forecast) in enumerate(future_forecasts.items()):
    fig_future.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines+markers', name=name,
        line=dict(color=colors[i], width=3),
        marker=dict(size=6)
    ))
    
    # Confidence bands for best model only
    if name == best_model_name:
        fig_future.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_future.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines', fill='tonexty', fillcolor=f'rgba({255 if i==0 else 0},{0 if i==1 else 255},{0 if i==2 else 255},0.2)',
            line=dict(width=0), name=f'{name} Confidence', showlegend=True
        ))

fig_future.update_layout(
    title="Future Forecast Comparison (Next 12 Months)",
    xaxis_title="Date", yaxis_title="Resale Price (SGD)",
    height=600, hovermode='x unified',
    yaxis=dict(tickformat='$,.0f')
)

st.plotly_chart(fig_future, use_container_width=True)

# Recommendation
st.subheader("🎯 Recommendation")

mape_best = performance[best_model_name]['MAPE']
if mape_best < 5:
    quality = "Excellent"
    color = "success"
elif mape_best < 10:
    quality = "Good"
    color = "info"
elif mape_best < 15:
    quality = "Fair"
    color = "warning"
else:
    quality = "Poor"
    color = "error"

st.markdown(f"""
### 🏆 **Recommended Model: {best_model_name}**

**Performance**: {quality} (MAPE: {mape_best:.1f}%)

**Why this model is best**:
- Lowest validation error (RMSE: ${performance[best_model_name]['RMSE']:,.0f})
- Mean prediction error: {mape_best:.1f}% of actual price
- Good balance of accuracy and reliability

**Next Steps**:
1. Use this model configuration in your main Streamlit app
2. Monitor performance over time
3. Retrain monthly with new data
""")

# Answer to the user's question
st.subheader("❓ Your Questions Answered")

st.markdown("""
**Q: Which model is best?**
A: Based on validation performance, **{best_model}** performs best with lowest prediction errors.

**Q: Do they forecast based on trained model?**
A: **YES!** All three options:
- ✅ Train on your historical data ({data_points} months)
- ✅ Learn patterns, trends, and seasonality from your data
- ✅ Use learned patterns to forecast future prices
- ✅ Are NOT using any pre-existing external model

The difference is in **how** they learn:
- **Option 1**: Learns only from time-based patterns (dates, seasonality)
- **Option 2**: Learns from time patterns + external factors (volatility, volume, age)
- **Option 3**: Learns from time patterns + key factors (age, volume)

All models are trained specifically on **your {town} - {flat_type}** data!
""".format(
    best_model=best_model_name,
    data_points=len(df_monthly),
    town=selected_town,
    flat_type=selected_flat
))

# Model saving section
if st.button("💾 Save Recommended Model"):
    recommended_model = models[best_model_name]
    
    # Determine regressors based on model type
    if 'Simple' in best_model_name:
        regressors_used = []
    elif 'High Flexibility' in best_model_name:
        regressors_used = ['price_volatility', 'transaction_volume', 'avg_age']
    else:
        regressors_used = ['avg_age', 'transaction_volume']
    
    # Create model package
    model_package = {
        'model': recommended_model,
        'regressors': regressors_used,
        'best_hyperparameters': {
            'changepoint_prior_scale': recommended_model.changepoint_prior_scale,
            'seasonality_prior_scale': recommended_model.seasonality_prior_scale
        },
        'training_period': {
            'start': df_monthly['ds'].min(),
            'end': df_monthly['ds'].max()
        },
        'model_version': f'recommended_{best_model_name.lower().replace(" ", "_").replace(":", "")}',
        'created_date': pd.Timestamp.now(),
        'forecasting_level': 'town_flat_type',
        'performance': performance[best_model_name],
        'production_ready': True,
        'last_known_values': {reg: df_monthly[reg].iloc[-1] for reg in regressors_used if reg in df_monthly.columns}
    }
    
    # Save model
    filename = f"prophet_recommended_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model_package, f)
    
    st.success(f"✅ Model saved as: {filename}")
    st.info("Replace your current 'prophet_optimized_model.pkl' with this file to use the best model!")

st.markdown("---")
st.markdown("💡 **Key Takeaway**: All models train on YOUR data - the difference is complexity vs. accuracy trade-off!")