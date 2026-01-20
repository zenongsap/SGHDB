
"""
📊 HDB Market Segmentation Precomputation
-----------------------------------------

Summary:
This script prepares a precomputed dataset for market segmentation analysis based on a trained hybrid pipeline.  
It applies feature engineering, scaling, dimensionality reduction, and clustering, then saves the results for later use 
in the Streamlit analytics app.

Steps Performed:
1. Load raw HDB resale data and sample 50K non-outlier records.
2. Load trained pipeline (scaler, selected features, UMAP reducer, and KMeans model).
3. Recreate engineered features:
   - Fill missing numeric values
   - Count encoding for categorical features
   - Row-level statistics (mean, std, range)
   - Log transformations (price per sqm, floor area)
4. Prepare feature matrix and apply scaling.
5. Apply UMAP dimensionality reduction and KMeans clustering.
6. Generate segmented dataset with cluster labels ("Segment_X").
7. Save results (dataframe + cluster info) into `precomputed_market.pkl`.

Prerequisites:
- Python 3.9+ (recommended)
- Install required packages:
    pip install pandas numpy scikit-learn umap-learn

- Required input files:
    • raw_data_main.csv                    → Raw HDB resale dataset
    • hybrid_multi_metric_pipeline_50K.pkl → Trained pipeline with scaler, reducer, and KMeans model

- Output file generated:
    • precomputed_market.pkl → Contains segmented dataframe, UMAP embeddings, cluster labels, and features list

Usage:
    Run the script locally with:
        python precompute_market.py
    (replace `precompute_market.py` with your filename)

"""


import pandas as pd
import numpy as np
import pickle

# -------------------------------
# 1️⃣ Load raw data & sample 50K
# -------------------------------
df = pd.read_csv("raw_data_main.csv")
df = df[df['IS_OUTLIERS'] != 1].sample(50000, random_state=42)

# -------------------------------
# 2️⃣ Load trained pipeline
# -------------------------------
with open("hybrid_multi_metric_pipeline_50K.pkl", "rb") as f:
    pipeline = pickle.load(f)

scaler = pipeline['scaler']
selected_features = pipeline['selected_features']
best_result = pipeline['best_result']
reducer = pipeline['reducer']
kmeans = best_result['kmeans']

# -------------------------------
# 3️⃣ Recreate engineered features
# -------------------------------
numeric_features = ['FLOOR_AREA_SQM', 'STOREY_NUMERIC', 'PRICE_PER_SQM', 'AGE']

# Fill missing numeric values
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Count encoding for categorical features
categorical_cols = ['TOWN', 'FLAT_TYPE']
for col in categorical_cols:
    counts = df[col].value_counts()
    df[f'{col}_COUNT'] = df[col].map(counts)

# Engineered features
df['ROW_MEAN'] = df[numeric_features].mean(axis=1)
df['ROW_STD'] = df[numeric_features].std(axis=1)
df['ROW_RANGE'] = df[numeric_features].max(axis=1) - df[numeric_features].min(axis=1)
df['PRICE_LOG'] = np.log1p(df['PRICE_PER_SQM'])
df['AREA_LOG'] = np.log1p(df['FLOOR_AREA_SQM'])

# -------------------------------
# 4️⃣ Prepare feature matrix & scale
# -------------------------------
X_all = df[selected_features].copy().fillna(0)
X_scaled = scaler.transform(X_all)

# -------------------------------
# 5️⃣ Apply UMAP + KMeans
# -------------------------------
X_umap = reducer.transform(X_scaled)
cluster_labels = kmeans.predict(X_umap)

# Keep all original columns needed for Streamlit
df_segmented = df.copy()
df_segmented['MARKET_SEGMENT'] = [f"Segment_{i}" for i in cluster_labels]

# Ensure DATE column is datetime
df_segmented['DATE'] = pd.to_datetime(
    df_segmented['YEAR'].astype(str) + '-' + df_segmented['MONTH_NUM'].astype(str) + '-01'
)


# -------------------------------
# 6️⃣ Save precomputed dataset
# -------------------------------
precomputed = {
    "df_segmented": df_segmented,
    "X_umap": X_umap,
    "cluster_labels": cluster_labels,
    "selected_features": selected_features
}

with open("precomputed_market.pkl", "wb") as f:
    pickle.dump(precomputed, f)