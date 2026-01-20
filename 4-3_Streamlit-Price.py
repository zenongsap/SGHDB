
"""
🏠 HDB Analytics Suite - Price Estimation
---------------------------------------------

Summary:
This Streamlit app provides 💰 Price Estimation of HDB resale flat to determine current flat fair valuation.
   - Uses a trained CatBoost machine learning model to estimate the resale price of an HDB flat based on user inputs  
     (Town, Flat Type, Floor Area, Storey Number, and Age).  
   - Provides insights such as price tier (Budget, Mid-range, Premium, Luxury), age group classification, and 
     comparisons against average and median prices for similar flats.  
   - Displays percentile ranking of the predicted price within comparable properties.  

Prerequisites:
- Python 3.9+ (recommended)  
- Install required packages:
    pip install streamlit pandas numpy prophet plotly scikit-learn catboost joblib

- Required data/model files in the same directory:
    • raw_data_main.csv                      → Raw HDB resale dataset  
    • 4-4_catboost_model_valid.pkl           → Trained CatBoost price prediction model  
    • 4-5_best_catboost_params.json          → Features & parameters for CatBoost model  

Usage:
    Run the app locally with:
        streamlit run app.py
    (replace `app.py` with your filename)

"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib
<<<<<<< Updated upstream
=======
import json
>>>>>>> Stashed changes
from catboost import Pool

# ====================
# 1. Load CatBoost Model & Features
# ====================
@st.cache_resource
def load_model_and_features():
<<<<<<< Updated upstream
    model = joblib.load("catboost_model_valid.pkl")
    features_used = joblib.load("CatBoost_features_used.pkl")  # List of columns used during training
=======
    model = joblib.load("4-4_catboost_model_valid.pkl")
    
    # Load features from the updated best_catboost_params.json
    with open("4-5_best_catboost_params.json", "r") as f:
        params_data = json.load(f)
    
    # Get the feature names from the updated params file
    features_used = params_data['features_used']
    
>>>>>>> Stashed changes
    return model, features_used

price_model, features_used = load_model_and_features()

# ====================
# 2. Load Raw Data (for reference columns & bins)
# ====================
@st.cache_data
def load_data():
    df_main = pd.read_csv("raw_data_main.csv")  
    return df_main

df_main = load_data()

# ====================
# 3. Price Prediction Function for CatBoost (Simplified)
# ====================

def predict_price_model(input_dict):
    input_df = pd.DataFrame([input_dict])
    
    # Compute AGE_GROUP for display
    bins_age = [0, 5, 15, 30, float('inf')]
    labels_age = ['New', 'Moderate', 'Old', 'Very Old']
    age_group = pd.cut(input_df['AGE'], bins=bins_age, labels=labels_age, right=False)[0]
    
    # -----------------------------
    # One-hot encode TOWN and FLAT_TYPE to match training
    # -----------------------------
    for col in features_used:
        if col.startswith('TOWN_'):
            input_df[col] = 1 if col == f"TOWN_{input_dict['TOWN']}" else 0
        elif col.startswith('FLAT_TYPE_'):
            input_df[col] = 1 if col == f"FLAT_TYPE_{input_dict['FLAT_TYPE']}" else 0
        elif col not in input_df.columns:
            input_df[col] = 0  # numeric features like FLOOR_AREA_SQM, STOREY_NUMERIC

    input_encoded = input_df[features_used]

    # -----------------------------
    # Predict price
    # -----------------------------
    price = price_model.predict(input_encoded)[0]

    # Convert back from log1p if your target was transformed
    price = np.expm1(price)

    # -----------------------------
    # Determine PRICE_TIER
    # -----------------------------
    price_bins = df_main['RESALE_PRICE'].quantile([0, 0.25, 0.75, 0.95, 1.0])
    labels_price = ['Budget', 'Mid-range', 'Premium', 'Luxury']
    price_tier = pd.cut([price], bins=price_bins, labels=labels_price, include_lowest=True)[0]

    return price, age_group, price_tier

# ====================
# Streamlit UI
# ====================
st.set_page_config(page_title="HDB Price Predictor", page_icon="🏠", layout="centered")
st.markdown('<div class="main-header">🏠 HDB Price Predictor</div>', unsafe_allow_html=True)

# Input form
st.markdown("### Estimate HDB Resale Price")
col1, col2 = st.columns(2)
with col1:
    est_town = st.selectbox("Select Town", sorted(df_main["TOWN"].unique()))
    est_flat_type = st.selectbox("Flat Type", df_main["FLAT_TYPE"].unique())
    est_floor_area = st.slider("Floor Area (sqm)", 60, 160, 90)
with col2:
    est_storey_num = st.slider("Storey Number", 1, 50, 10)  # Use numeric directly
<<<<<<< Updated upstream
    est_age = st.slider("Flat Age (years)", 0, 50, 15)
=======
    est_age = st.slider("Flat Age (years)", 0, 60, 15)
>>>>>>> Stashed changes

# Prediction button
if st.button("🔍 Estimate Price"):
    input_dict = {
        "FLOOR_AREA_SQM": est_floor_area,
        "STOREY_NUMERIC": est_storey_num,
        "AGE": est_age,
        "TOWN": est_town,
        "FLAT_TYPE": est_flat_type
    }

    estimated_price, age_group, price_tier = predict_price_model(input_dict)
    
    # Tier colors
    tier_colors = {
        'Budget': "#27ae60",
        'Mid-range': "#f39c12",
        'Premium': "#e74c3c",
        'Luxury': "#9b59b6"
    }
    
    # Determine PRICE_TIER as string
    price_tier_str = str(price_tier) if pd.notna(price_tier) else "Mid-range"  # fallback

    # Get tier color safely
    tier_color = tier_colors.get(price_tier_str, "#95a5a6")  # gray as default
    
    st.markdown(f"""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {tier_color}20, {tier_color}10); 
            border-radius: 15px; border: 2px solid {tier_color};">
    <h2 style="color: {tier_color}; margin: 0;">💰 SGD {estimated_price:,.0f}</h2>
    <p style="color: {tier_color}; font-weight: bold; margin: 5px 0;">{price_tier_str} Segment</p>
    <p style="color: #666; margin: 0; font-size: 14px;">Age Group: {age_group}</p>
    <p style="color: #666; margin: 0; font-size: 14px;">{est_flat_type} in {est_town} • {est_floor_area}sqm • {est_age} years old</p>
</div>
<<<<<<< Updated upstream
""", unsafe_allow_html=True)
=======
""", unsafe_allow_html=True)
>>>>>>> Stashed changes
