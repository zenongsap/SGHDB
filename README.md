# 🏠 HDB Analytics Suite

A comprehensive Streamlit-based toolkit for analyzing, forecasting, and segmenting Singapore HDB resale prices.  
This suite allows users to explore historical resale data, predict prices, forecast trends, and understand market segments through precomputed clustering.

---
## Project Overview

HDB Analytics Suite consists of **three main components**:

1. **Price Estimation & Price Forecasting** (`Streamlit-PriceNForecast.py`)  
   - **💰 Price Estimation:** Predict resale price for a specific HDB flat based on user input:
     - Inputs: Town, Flat Type, Floor Area, Storey Number, Flat Age  
     - Output: Estimated price, price tier (Budget / Mid-range / Premium / Luxury), age group, percentile rank, and comparison with average/median prices
   
   - **🔮 Price Forecasting:** Predict future resale prices for a selected Town and Flat Type using a Prophet time-series model
     - Inputs: Town, Flat Type, Forecast Period (6-24 months)  
     - Output: Historical and forecasted price charts with confidence intervals, forecast summary tables, and trend metrics

2. **Market Segmentation Dashboard** (`5-5_Streamlit-market.py`)  
   - Visualizes the precomputed HDB market segmentation results
   - Features:
     - Filters by Town, Flat Type, Age Group, Price Range, Floor Area
     - Metrics: Total Properties, Average Price, Towns & Flat Types Count
     - Visualizations:
       - Flat Age Distribution by Town
       - Flat Type Distribution by Town
       - Price Heatmap (Resale Price vs Price per sqm)
       - Resale Price Trend Over Time
     - Segment Summary Table with Mean/Median Price, Property Count, Average Area & Age

Other supporting file (NOT CORE)
1. Market Segmentation Precomputation ** (`5-2_precompute_market.py`)  
   - Prepares precomputed dataset for market segmentation analysis based on a trained hybrid pipeline (UMAP + KMeans)
   - Performs:
     - Feature engineering: numeric fill, count encoding, row statistics, log transformations  
     - Scaling, dimensionality reduction, clustering  
     - Adds cluster labels as `MARKET_SEGMENT` for each property
   - Output: `precomputed_market.pkl` containing segmented dataframe, UMAP embeddings, cluster labels, and features

2. Model comparison for Forecasting ('6-4_Streamlit-ML-Forecast-tool.py')
   - Three Prophet models are trained on the historical data:
     Option 1: Simple Model (time-based only)
     Option 2: High Flexibility (time + regressors: volatility, volume, age)
     Option 3: Key Regressors (time + regressors: age, volume)
   - Models are evaluated on last 20% of data for validation using MAE, RMSE, and MAPE.
   - Best model (lowest RMSE) is highlighted.
-

## Installation

1. Clone this repository:

```bash
git clone --branch main --single-branch https://gitlab.com/nyp-sg/cet/shc-c-nyp-sit-sctp/daai-intake1/c2-learner03.git
cd c2-learner03

2. Install required packages:
pip install streamlit pandas numpy plotly scikit-learn prophet catboost joblib

3. Required files :

	File Name					Purpose
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	raw_data_main.csv				Raw HDB resale dataset
	4-4_catboost_model_valid.pkl			CatBoost model for price estimation
	4-5_best_catboost_params.json			Features & parameters for CatBoost model
	6-4_prophet_full_pipeline.pkl			Trained Prophet model for price forecasting
	5-3_hybrid_multi_metric_pipeline_50K.pkl	Trained pipeline for market segmentation precomputation
	5-4_precomputed_market.pkl			Output of precomputation script used in segmentation dashboard


## Usage

1️⃣ Price Estimation & Forecasting

	Run the app with:
	streamlit run Streamlit-PriceNForecast.py

   * Use sidebar to switch between Price Estimation and Price Forecasting
   * Input flat details or select town/flat type for forecasting
   * View metrics, interactive charts, and forecast summaries


2️⃣ Market Segmentation Dashboard
	Run the app with:
	streamlit run 5-5_Streamlit-market.py

   * It is using 50K Precomputed data ensures fast performance by avoiding real-time clustering computation


This suite is for informational purposes only and should not be used as the sole basis for investment decisions