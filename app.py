"""
🏠 HDB Analytics Suite (Unified App)
-----------------------------------

This Streamlit app combines:
- Streamlit-PriceNForecast.py (Price Estimation + Price Forecasting)
- 5-5_Streamlit-market.py (Market Segmentation Dashboard)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from prophet import Prophet

warnings.filterwarnings("ignore")


HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

MOBILE_CSS = """
<style>
@media (max-width: 768px) {
  .block-container {
    padding-top: 0.5rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
  }

  button[kind="primary"], button[kind="secondary"] {
    padding-top: 0.6rem !important;
    padding-bottom: 0.6rem !important;
    font-size: 0.9rem !important;
  }

  .stMetric {
    text-align: center !important;
  }

  .element-container iframe, .element-container img {
    max-width: 100% !important;
    height: auto !important;
  }
}
</style>
"""


# ====================
# Shared helpers
# ====================
def format_to_k(price: float) -> str:
    """Formats a numeric price value to a string like '$349K'."""
    if pd.isna(price) or price == 0:
        return "$0K"
    return f"${price/1000:.0f}K"


def assign_age_group_label(age: float) -> str:
    """Market app age-group labels (kept identical for compatibility)."""
    if age < 5:
        return "New(Below 5yrs)"
    if age < 15:
        return "Moderate(5-14yrs)"
    if age < 30:
        return "Old(15-29yrs)"
    return "Very Old(Above 30yrs)"


# ====================
# Data / model loading
# ====================
@st.cache_data
def load_raw_main_data() -> pd.DataFrame:
    df = pd.read_csv("raw_data_main.csv")
    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MONTH_NUM"]].rename(columns={"MONTH_NUM": "month"}).assign(day=1)
    )
    return df


@st.cache_resource
def load_price_model_and_features():
    model = joblib.load("4-4_catboost_model_valid.pkl")
    with open("4-5_best_catboost_params.json", "r", encoding="utf-8") as f:
        params_data = json.load(f)
    features_used = params_data["features_used"]
    return model, features_used


@st.cache_resource
def load_forecast_pipeline():
    with open("6-4_prophet_full_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


@st.cache_data
def load_precomputed_market_df():
    with open("5-4_precomputed_market.pkl", "rb") as f:
        data = pickle.load(f)
    return data["df_segmented"]


# ====================
# Price estimation logic
# ====================
def predict_price_model(input_dict, price_model, features_used, df_main):
    input_df = pd.DataFrame([input_dict])

    # Compute AGE_GROUP for display (kept identical to original)
    bins_age = [0, 5, 15, 30, float("inf")]
    labels_age = ["New", "Moderate", "Old", "Very Old"]
    age_group = pd.cut(
        input_df["AGE"], bins=bins_age, labels=labels_age, right=False
    )[0]

    # One-hot encode TOWN and FLAT_TYPE to match training
    for col in features_used:
        if col.startswith("TOWN_"):
            input_df[col] = 1 if col == f"TOWN_{input_dict['TOWN']}" else 0
        elif col.startswith("FLAT_TYPE_"):
            input_df[col] = 1 if col == f"FLAT_TYPE_{input_dict['FLAT_TYPE']}" else 0
        elif col not in input_df.columns:
            input_df[col] = 0

    input_encoded = input_df[features_used]
    price = price_model.predict(input_encoded)[0]
    price = np.expm1(price)

    # Determine PRICE_TIER (kept identical to original)
    price_bins = df_main["RESALE_PRICE"].quantile([0, 0.25, 0.75, 0.95, 1.0])
    labels_price = ["Budget", "Mid-range", "Premium", "Luxury"]
    price_tier = pd.cut([price], bins=price_bins, labels=labels_price, include_lowest=True)[
        0
    ]

    return price, age_group, price_tier


# ====================
# Market segmentation charts (ported as-is)
# ====================
def create_flat_age_distribution(df, selected_towns=None):
    if selected_towns:
        df_filtered = df[df["TOWN"].isin(selected_towns)]
    else:
        df_filtered = df.copy()
    age_group_counts = (
        df_filtered.groupby(["TOWN", "AGE_GROUP"]).size().unstack(fill_value=0)
    )
    age_colors = {
        "New(Below 5yrs)": "#A6CEE3",
        "Moderate(5-14yrs)": "#1F78B4",
        "Old(15-29yrs)": "#FDBF6F",
        "Very Old(Above 30yrs)": "#F4A6A6",
    }
    fig = go.Figure()
    for group in age_group_counts.columns:
        fig.add_trace(
            go.Bar(
                name=group,
                x=age_group_counts.index,
                y=age_group_counts[group],
                marker_color=age_colors.get(group, "#808080"),
            )
        )
    fig.update_layout(
        title="Flat Age Distribution by Town",
        xaxis_title="Town",
        yaxis_title="Number of Flats",
        barmode="stack",
        height=500,
        showlegend=True,
    )
    return fig


def create_price_timeline(df, selected_towns=None, selected_flat_types=None):
    if selected_towns:
        df = df[df["TOWN"].isin(selected_towns)]
    if selected_flat_types:
        df = df[df["FLAT_TYPE"].isin(selected_flat_types)]
    timeline_data = df.groupby("DATE")["RESALE_PRICE"].agg(["mean", "count"]).reset_index()
    timeline_data = timeline_data[timeline_data["count"] >= 5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline_data["DATE"],
            y=timeline_data["mean"],
            mode="lines+markers",
            name="Mean Resale Price",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title="Resale Price Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Mean Resale Price (SGD)",
        height=400,
    )
    return fig


def create_flat_type_distribution(df, selected_towns=None):
    if selected_towns:
        df_filtered = df[df["TOWN"].isin(selected_towns)]
    else:
        df_filtered = df
    flat_type_counts = (
        df_filtered.groupby(["TOWN", "FLAT_TYPE"]).size().unstack(fill_value=0)
    )
    flat_type_colors = {
        "3 ROOM": "#B3CDE3",
        "4 ROOM": "#6497B1",
        "5 ROOM": "#FFA07A",
        "EXECUTIVE": "#CD5C5C",
    }
    fig = go.Figure()
    for flat_type in flat_type_counts.columns:
        fig.add_trace(
            go.Bar(
                name=flat_type,
                x=flat_type_counts.index,
                y=flat_type_counts[flat_type],
                marker_color=flat_type_colors.get(flat_type, "#808080"),
            )
        )
    fig.update_layout(
        title="Flat Type Distribution by Town",
        xaxis_title="Town",
        yaxis_title="Number of Properties",
        barmode="stack",
        height=500,
        showlegend=True,
    )
    return fig


def create_price_heatmap(df, selected_towns=None, selected_flat_types=None):
    if selected_towns:
        df = df[df["TOWN"].isin(selected_towns)]
    if selected_flat_types:
        df = df[df["FLAT_TYPE"].isin(selected_flat_types)]
    resale_data = (
        df.groupby(["TOWN", "FLAT_TYPE"])["RESALE_PRICE"].mean().unstack(fill_value=0)
        / 1000
    )
    sqm_data = df.groupby(["TOWN", "FLAT_TYPE"])["PRICE_PER_SQM"].mean().unstack(fill_value=0)
    custom_colorscale = [[0.0, "#e6f2ff"], [1.0, "#003366"]]
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=resale_data.values,
            x=resale_data.columns,
            y=resale_data.index,
            colorscale=custom_colorscale,
            text=resale_data.round(0).astype(int),
            texttemplate="%{text}K",
            textfont={"size": 10},
            colorbar=dict(title="Price (SGD K)"),
            visible=True,
            name="Resale Price",
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=sqm_data.values,
            x=sqm_data.columns,
            y=sqm_data.index,
            colorscale=custom_colorscale,
            text=sqm_data.round(0).astype(int),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Price per sqm (SGD)"),
            visible=False,
            name="Price per sqm",
        )
    )
    fig.update_layout(
        title="Mean Price Heatmap (Town vs Flat Type)",
        xaxis_title="Flat Type",
        yaxis_title="Town",
        height=600,
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Resale Price (K SGD)",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {"title": "Mean Resale Price Heatmap"},
                        ],
                    ),
                    dict(
                        label="Price per sqm (SGD)",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {"title": "Mean Price per sqm Heatmap"},
                        ],
                    ),
                ],
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    return fig


def create_price_boxplot(df, selected_flat_types=None):
    if selected_flat_types:
        df = df[df["FLAT_TYPE"].isin(selected_flat_types)]

    flat_type_order = ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
    df = df.copy()
    df["FLAT_TYPE"] = pd.Categorical(
        df["FLAT_TYPE"], categories=flat_type_order, ordered=True
    )

    fig = px.box(
        df,
        x="TOWN",
        y="RESALE_PRICE",
        color="FLAT_TYPE",
        category_orders={"FLAT_TYPE": flat_type_order},
        title="Resale Price Distribution by Town and Flat Type",
        labels={"RESALE_PRICE": "Resale Price (SGD)", "TOWN": "Town"},
        height=500,
    )
    fig.update_layout(boxmode="group", xaxis_tickangle=-45, showlegend=True)
    return fig


# ====================
# Page renderers
# ====================
def render_price_estimation(df: pd.DataFrame) -> None:
    st.title("💰 HDB Resale Price Estimation")
    st.markdown("Get instant price estimates for HDB resale flats using machine learning")

    try:
        price_model, features_used = load_price_model_and_features()
    except Exception as e:
        st.error(f"Error loading price model: {e}")
        return

    st.subheader("🏠 Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        est_town = st.selectbox(
            "Select Town", sorted(df["TOWN"].unique()), key="est_town_select"
        )
        est_flat_type = st.selectbox(
            "Flat Type", df["FLAT_TYPE"].unique(), key="est_flat_type_select"
        )
        est_floor_area = st.slider("Floor Area (sqm)", 60, 160, 90)
    with col2:
        est_storey_num = st.slider("Storey Number", 1, 50, 10)
        est_age = st.slider("Flat Age (years)", 0, 60, 15)

    st.markdown("---")

    if st.button("🔍 Estimate Price", type="primary"):
        input_dict = {
            "FLOOR_AREA_SQM": est_floor_area,
            "STOREY_NUMERIC": est_storey_num,
            "AGE": est_age,
            "TOWN": est_town,
            "FLAT_TYPE": est_flat_type,
        }

        estimated_price, age_group, price_tier = predict_price_model(
            input_dict, price_model, features_used, df
        )

        tier_colors = {
            "Budget": "#27ae60",
            "Mid-range": "#f39c12",
            "Premium": "#e74c3c",
            "Luxury": "#9b59b6",
        }

        price_tier_str = str(price_tier) if pd.notna(price_tier) else "Mid-range"
        tier_color = tier_colors.get(price_tier_str, "#95a5a6")

        st.markdown(
            f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {tier_color}20, {tier_color}10);
                        border-radius: 20px; border: 3px solid {tier_color}; margin: 20px 0;">
                <h1 style="color: {tier_color}; margin: 0; font-size: 48px;">💰 SGD {estimated_price:,.0f}</h1>
                <h3 style="color: {tier_color}; font-weight: bold; margin: 10px 0;">{price_tier_str} Segment</h3>
                <p style="color: #666; margin: 5px 0; font-size: 16px;">Age Group: {age_group}</p>
                <p style="color: #666; margin: 0; font-size: 16px;">{est_flat_type} in {est_town} • {est_floor_area}sqm • Storey {est_storey_num} • {est_age} years old</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("📊 Price Analysis")

        col1, col2, col3 = st.columns(3)
        comparable_data = df[(df["TOWN"] == est_town) & (df["FLAT_TYPE"] == est_flat_type)]

        if not comparable_data.empty:
            avg_price = comparable_data["RESALE_PRICE"].mean()
            median_price = comparable_data["RESALE_PRICE"].median()

            with col1:
                diff_from_avg = ((estimated_price - avg_price) / avg_price) * 100
                st.metric("vs Average", format_to_k(avg_price), delta=f"{diff_from_avg:+.1f}%")
            with col2:
                diff_from_median = ((estimated_price - median_price) / median_price) * 100
                st.metric(
                    "vs Median", format_to_k(median_price), delta=f"{diff_from_median:+.1f}%"
                )
            with col3:
                percentile = (comparable_data["RESALE_PRICE"] <= estimated_price).mean() * 100
                st.metric("Price Percentile", f"{percentile:.0f}%")

    with st.expander("📋 View Sample Data"):
        sample_data = df.sample(10)[
            ["TOWN", "FLAT_TYPE", "FLOOR_AREA_SQM", "STOREY_NUMERIC", "AGE", "RESALE_PRICE"]
        ]
        sample_data = sample_data.copy()
        sample_data["RESALE_PRICE"] = sample_data["RESALE_PRICE"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(sample_data, use_container_width=True)


def render_price_forecasting(df: pd.DataFrame) -> None:
    st.title("🔮 HDB Resale Price Forecasting")
    st.markdown("Predict future HDB resale prices using advanced time series analysis")

    try:
        pipeline = load_forecast_pipeline()
        prophet_model = pipeline["model"]
    except Exception as e:
        st.error(f"Error loading forecast model: {e}")
        return

    # In-content, mobile-friendly filters
    st.markdown("#### Forecast Settings")
    col1, col2, col3 = st.columns(3)

    towns = sorted(df["TOWN"].unique())
    flat_types = sorted(df["FLAT_TYPE"].unique())
    forecast_options = [6, 12, 18, 24]

    with col1:
        selected_town = st.selectbox("Town", towns, key="forecast_town_select")
    with col2:
        selected_flat = st.selectbox("Flat Type", flat_types, key="forecast_flat_type_select")
    with col3:
        forecast_period = st.selectbox(
            "Period (Months)", forecast_options, index=1, key="forecast_period_select"
        )

    df_filtered = df[(df["TOWN"] == selected_town) & (df["FLAT_TYPE"] == selected_flat)].copy()
    if df_filtered.empty:
        st.warning("No data found for this Town and Flat Type combination.")
        return

    agg_cols = {
        "RESALE_PRICE": ["mean", "median", "count", "std"],
        "FLOOR_AREA_SQM": "mean",
        "AGE": "mean",
        "STOREY_NUMERIC": "mean",
        "IS_OUTLIERS": "sum",
    }
    df_monthly = df_filtered.groupby(["DATE"]).agg(agg_cols).round(2)
    df_monthly.columns = ["_".join(col).strip() for col in df_monthly.columns.values]
    df_monthly = df_monthly.reset_index()
    df_monthly = df_monthly.rename(columns={"DATE": "ds", "RESALE_PRICE_mean": "y"})
    df_monthly = df_monthly.sort_values("ds").reset_index(drop=True)

    st.subheader(f"📈 Historical Data Overview: {selected_town} - {selected_flat}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Price", format_to_k(df_monthly["y"].mean()))
    with col2:
        st.metric("Min Price", format_to_k(df_monthly["y"].min()))
    with col3:
        st.metric("Max Price", format_to_k(df_monthly["y"].max()))
    with col4:
        st.metric("Data Points", len(df_monthly))

    local_model = Prophet(
        changepoint_prior_scale=prophet_model.changepoint_prior_scale,
        seasonality_prior_scale=prophet_model.seasonality_prior_scale,
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
    )
    local_model.fit(df_monthly)

    future = local_model.make_future_dataframe(periods=forecast_period, freq="MS")
    forecast = local_model.predict(future)
    forecast_merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        df_monthly, on="ds", how="left"
    )

    historical_data = forecast_merged[forecast_merged["y"].notna()]
    future_data = forecast_merged[forecast_merged["y"].isna()]

    st.subheader("📊 Forecasted Resale Prices")
    min_year = int(forecast_merged["ds"].dt.year.min())
    max_year = int(forecast_merged["ds"].dt.year.max())
    start_year, end_year = st.slider(
        "Select Year Range to Display:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    chart_hist_data = historical_data[
        (historical_data["ds"].dt.year >= start_year)
        & (historical_data["ds"].dt.year <= end_year)
    ]
    chart_future_data = future_data[
        (future_data["ds"].dt.year >= start_year) & (future_data["ds"].dt.year <= end_year)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_hist_data["ds"],
            y=chart_hist_data["y"],
            mode="lines+markers",
            name="Historical Actual",
            line=dict(color="black", width=2),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_hist_data["ds"],
            y=chart_hist_data["yhat"],
            mode="lines",
            name="Historical Fitted",
            line=dict(color="blue", width=1, dash="dot"),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_future_data["ds"],
            y=chart_future_data["yhat"],
            mode="lines+markers",
            name="Future Forecast",
            line=dict(color="red", width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_future_data["ds"],
            y=chart_future_data["yhat_upper"],
            mode="lines",
            name="Confidence Upper",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_future_data["ds"],
            y=chart_future_data["yhat_lower"],
            mode="lines",
            name="Confidence Band",
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(255, 0, 0, 0.2)",
            showlegend=True,
        )
    )
    fig.update_layout(
        title=f"Resale Price Forecast: {selected_town} - {selected_flat}",
        xaxis_title="Date",
        yaxis_title="Resale Price (SGD)",
        hovermode="x unified",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔮 Forecast Summary")
        if not future_data.empty:
            summary_table = future_data[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            summary_table["ds"] = summary_table["ds"].dt.strftime("%Y-%m")
            summary_table.columns = ["Month", "Forecast", "Conf_Lower", "Conf_Upper"]
            for col in ["Forecast", "Conf_Lower", "Conf_Upper"]:
                summary_table[col] = summary_table[col].apply(format_to_k)
            st.dataframe(summary_table, use_container_width=True)
        else:
            st.warning("No future forecasts available.")
    with col2:
        st.subheader("💡 Key Insights")
        if not future_data.empty and not historical_data.empty:
            latest_actual = historical_data["y"].iloc[-1]
            last_forecast = future_data["yhat"].iloc[-1]
            price_change = last_forecast - latest_actual
            price_change_pct = (price_change / latest_actual * 100) if latest_actual > 0 else 0

            st.metric("Current Price", format_to_k(latest_actual))
            st.metric(
                f"Forecast ({forecast_period}M)",
                format_to_k(last_forecast),
                delta=f"{price_change_pct:+.1f}%",
            )
            trend = (
                "📈 Upward" if price_change > 0 else "📉 Downward" if price_change < 0 else "➡️ Stable"
            )
            st.metric("Trend", trend)


def render_market_segmentation() -> None:
    st.title("📊 Market Segmentation")

    try:
        df = load_precomputed_market_df()
    except FileNotFoundError:
        st.error("❌ File not found: 5-4_precomputed_market.pkl")
        return
    except Exception as e:
        st.error(f"❌ Error loading precomputed data: {str(e)}")
        st.exception(e)
        return

    if "AGE_GROUP" not in df.columns:
        df = df.copy()
        df["AGE_GROUP"] = df["AGE"].apply(assign_age_group_label)

    st.markdown("#### Analysis Filters")
    available_towns = sorted(df["TOWN"].unique())
    available_flat_types = sorted(df["FLAT_TYPE"].unique())
    available_age_groups = sorted(df["AGE_GROUP"].unique())

    col1, col2 = st.columns(2)
    with col1:
        selected_towns = st.multiselect(
            "Towns",
            available_towns,
            default=available_towns[:5],
            key="market_towns_multiselect",
        )
        selected_age_groups = st.multiselect(
            "Age Groups",
            available_age_groups,
            default=available_age_groups,
            key="market_age_groups_multiselect",
        )
    with col2:
        selected_flat_types = st.multiselect(
            "Flat Types",
            available_flat_types,
            default=available_flat_types,
            key="market_flat_types_multiselect",
        )

        min_price, max_price = int(df["RESALE_PRICE"].min()), int(df["RESALE_PRICE"].max())
        price_range = st.slider(
            "Price Range (SGD)",
            min_price,
            max_price,
            (min_price, max_price),
            step=1000,
            key="market_price_slider",
        )

        min_area, max_area = int(df["FLOOR_AREA_SQM"].min()), int(df["FLOOR_AREA_SQM"].max())
        area_range = st.slider(
            "Floor Area (sqm)",
            min_area,
            max_area,
            (min_area, max_area),
            step=5,
            key="market_area_slider",
        )

    df_filtered = df[
        (df["TOWN"].isin(selected_towns))
        & (df["FLAT_TYPE"].isin(selected_flat_types))
        & (df["AGE_GROUP"].isin(selected_age_groups))
        & (df["RESALE_PRICE"].between(price_range[0], price_range[1]))
        & (df["FLOOR_AREA_SQM"].between(area_range[0], area_range[1]))
    ]

    st.subheader("📈 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df_filtered):,}")
    with col2:
        st.metric("Avg Price", format_to_k(df_filtered["RESALE_PRICE"].mean()))
    with col3:
        st.metric("Towns Analyzed", len(selected_towns))
    with col4:
        st.metric("Flat Types", len(selected_flat_types))

    st.markdown("---")
    col1, col2 = st.columns(2)

    st.subheader("📊 Price Distribution by Town and Flat Type")
    st.plotly_chart(create_price_boxplot(df_filtered, selected_flat_types), use_container_width=True)

    with col1:
        st.subheader("💰 Flat Age Distribution")
        st.plotly_chart(
            create_flat_age_distribution(df_filtered, selected_towns), use_container_width=True
        )
    with col2:
        st.subheader("🏠 Flat Type Distribution")
        st.plotly_chart(
            create_flat_type_distribution(df_filtered, selected_towns), use_container_width=True
        )

    st.subheader("🌡️ Price Heatmap")
    st.plotly_chart(
        create_price_heatmap(df_filtered, selected_towns, selected_flat_types),
        use_container_width=True,
    )

    st.subheader("📈 Price Trend Over Time")
    st.plotly_chart(
        create_price_timeline(df_filtered, selected_towns, selected_flat_types),
        use_container_width=True,
    )

    if "MARKET_SEGMENT" in df_filtered.columns:
        st.markdown("---")
        st.subheader("🎯 Market Segments Analysis")
        segment_summary = (
            df_filtered.groupby("MARKET_SEGMENT")
            .agg(
                {
                    "RESALE_PRICE": ["mean", "median", "count"],
                    "FLOOR_AREA_SQM": "mean",
                    "AGE": "mean",
                }
            )
            .round(2)
        )
        segment_summary.columns = ["Mean_Price", "Median_Price", "Count", "Avg_Area", "Avg_Age"]
        segment_summary = segment_summary.reset_index()
        segment_summary["Mean_Price_K"] = segment_summary["Mean_Price"].apply(format_to_k)
        segment_summary["Median_Price_K"] = segment_summary["Median_Price"].apply(format_to_k)
        st.dataframe(
            segment_summary[
                ["MARKET_SEGMENT", "Count", "Mean_Price_K", "Median_Price_K", "Avg_Area", "Avg_Age"]
            ],
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="HDB Analytics Suite",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    st.markdown("### 🏠 HDB Analytics Suite")
    st.caption("Price estimation, price forecasting, and market segmentation in one place.")

    # Tabs as primary navigation (no sidebar dependency)
    tab_estimate, tab_forecast, tab_market = st.tabs(
        ["💰 Price Estimation", "🔮 Price Forecasting", "📊 Market Segmentation"]
    )

    # Shared data for price / forecast
    df_main = None
    try:
        df_main = load_raw_main_data()
    except FileNotFoundError:
        st.error("❌ File not found: raw_data_main.csv")
    except Exception as e:
        st.error(f"❌ Error loading raw_data_main.csv: {e}")

    with tab_estimate:
        if df_main is not None:
            render_price_estimation(df_main)

    with tab_forecast:
        if df_main is not None:
            render_price_forecasting(df_main)

    with tab_market:
        render_market_segmentation()

    st.markdown("---")
    st.markdown(
        "*This application is for informational purposes only and should not be used as the sole basis for investment decisions.*"
    )


if __name__ == "__main__":
    main()

