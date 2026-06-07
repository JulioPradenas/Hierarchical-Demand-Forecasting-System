from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# App configuration
st.set_page_config(
    page_title="Hierarchical Demand Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-success {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Forecast Explorer", "Hierarchy Viewer", "Reconciliation Analysis"],
    index=0,
)


# Load placeholder data
@st.cache_data
def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample forecast and hierarchy data."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")

    # Sample forecast data
    forecast = pd.DataFrame(
        {
            "date": list(dates) + list(dates[:28]),
            "item_id": (["FOODS_1_001"] * 100) + (["FOODS_1_001"] * 28),
            "store_id": (["CA_1"] * 100) + (["CA_1"] * 28),
            "actual": list(np.random.poisson(50, 100)) + [np.nan] * 28,
            "forecast": list(50 + np.random.normal(0, 5, 100))
            + list(50 + np.random.normal(0, 5, 28)),
            "lower_80": list(40 + np.random.normal(0, 5, 100))
            + list(40 + np.random.normal(0, 5, 28)),
            "upper_80": list(60 + np.random.normal(0, 5, 100))
            + list(60 + np.random.normal(0, 5, 28)),
            "lower_95": list(30 + np.random.normal(0, 5, 100))
            + list(30 + np.random.normal(0, 5, 28)),
            "upper_95": list(70 + np.random.normal(0, 5, 100))
            + list(70 + np.random.normal(0, 5, 28)),
        }
    )

    # Sample hierarchy data
    hierarchy = pd.DataFrame(
        {
            "item_id": ["FOODS_1_001", "FOODS_1_002", "FOODS_2_001"],
            "category": ["FOODS_1", "FOODS_1", "FOODS_2"],
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "state": ["CA", "CA", "CA"],
            "wrmsse": [0.45, 0.52, 0.38],
            "coherence_error": [0.02, 0.01, 0.03],
        }
    )

    return forecast, hierarchy


# Page 1: Forecast Explorer
if page == "Forecast Explorer":
    st.title("📈 Forecast Explorer")
    st.markdown("Explore forecasts for individual items with reconciliation toggle.")

    forecast_df, hierarchy_df = load_sample_data()

    # Selectors
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        item_id = st.selectbox(
            "Select Item:",
            hierarchy_df["item_id"].unique(),
            index=0,
        )
    with col2:
        store_id = st.selectbox(
            "Select Store:",
            hierarchy_df["store_id"].unique(),
            index=0,
        )
    with col3:
        show_reconciled = st.checkbox("Show Reconciled Forecast", value=True)

    # Filter data
    mask = (forecast_df["item_id"] == item_id) & (forecast_df["store_id"] == store_id)
    data = forecast_df[mask].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    # Split historical and forecast
    historical = data[data["actual"].notna()].copy()
    future = data[data["actual"].isna()].copy()

    # Chart
    st.subheader(f"{item_id} @ {store_id}")

    chart_cols = st.columns([3, 1])
    with chart_cols[0]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        # Historical data
        if len(historical) > 0:
            ax.plot(
                historical["date"],
                historical["actual"],
                "ko-",
                label="Actual",
                linewidth=2,
            )

        # Forecast
        ax.plot(
            historical["date"],
            historical["forecast"],
            "b-",
            label="Forecast",
            linewidth=1.5,
        )

        # Intervals
        ax.fill_between(
            historical["date"],
            historical["lower_95"],
            historical["upper_95"],
            alpha=0.2,
            label="95% Interval",
        )
        ax.fill_between(
            historical["date"],
            historical["lower_80"],
            historical["upper_80"],
            alpha=0.3,
            label="80% Interval",
        )

        # Future forecast
        if len(future) > 0:
            future_dates = future["date"].values
            ax.plot(
                future_dates,
                future["forecast"],
                "b--",
                label="Future Forecast",
                linewidth=1.5,
            )
            ax.fill_between(
                future_dates,
                future["lower_95"],
                future["upper_95"],
                alpha=0.2,
            )
            ax.fill_between(
                future_dates,
                future["lower_80"],
                future["upper_80"],
                alpha=0.3,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.set_title("Demand Forecast with Prediction Intervals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    with chart_cols[1]:
        st.metric("Coherence Error", "0.015", "-0.005")
        st.metric("WRMSSE", "0.42", "-0.08")
        st.metric("Forecast Horizon", "28 days")

# Page 2: Hierarchy Viewer
elif page == "Hierarchy Viewer":
    st.title("🌳 Hierarchy Viewer")
    st.markdown("Interactive view of the hierarchical structure with forecast metrics.")

    forecast_df, hierarchy_df = load_sample_data()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Hierarchy Levels")

        # Aggregation levels
        levels = ["Item-Store", "Category-Store", "State", "Total"]
        selected_level = st.radio("Select Level:", levels, index=0)

        st.subheader("Metrics by Level")

        metrics_data = {
            "Level": ["Item-Store", "Category-Store", "State", "Total"],
            "Count": [3, 2, 1, 1],
            "Avg WRMSSE": [0.45, 0.42, 0.40, 0.38],
            "Avg Coherence Error": [0.02, 0.015, 0.01, 0.005],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader(f"Forecasts @ {selected_level}")

        # Display forecasts
        display_data = hierarchy_df[
            ["item_id", "category", "wrmsse", "coherence_error"]
        ].head(5)
        display_data.columns = ["Item", "Category", "WRMSSE", "Coherence Error"]

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        # Color-coded by uncertainty
        st.subheader("Uncertainty Heatmap")

        heatmap_data = pd.DataFrame(
            {
                "Item": ["FOODS_1_001", "FOODS_1_002", "FOODS_2_001"],
                "CA": [0.45, 0.52, 0.38],
                "TX": [0.48, 0.55, 0.40],
                "WA": [0.42, 0.50, 0.39],
            }
        )

        st.dataframe(heatmap_data, use_container_width=True, hide_index=True)

# Page 3: Reconciliation Analysis
else:  # Reconciliation Analysis
    st.title("🔄 Reconciliation Analysis")
    st.markdown("Compare reconciliation methods and their impact on coherence.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reconciliation Methods")

        methods = {
            "Bottom-Up": {"coherence_error": 0.001, "wrmsse": 0.55, "color": "🟢"},
            "Top-Down": {"coherence_error": 0.005, "wrmsse": 0.45, "color": "🟡"},
            "MinT": {"coherence_error": 0.002, "wrmsse": 0.42, "color": "🔵"},
            "OLS": {"coherence_error": 0.003, "wrmsse": 0.44, "color": "🟣"},
        }

        method_data = []
        for method, metrics in methods.items():
            method_data.append(
                {
                    "Method": method,
                    "Coherence Error": f"{metrics['coherence_error']:.4f}",
                    "WRMSSE": f"{metrics['wrmsse']:.2f}",
                }
            )

        st.dataframe(
            pd.DataFrame(method_data), use_container_width=True, hide_index=True
        )

    with col2:
        st.subheader("Business Impact")

        impact_data = pd.DataFrame(
            {
                "Metric": [
                    "Total Cost Reduction",
                    "Safety Stock Improvement",
                    "Service Level",
                ],
                "Value": ["12.5%", "8.3%", "94.2%"],
                "Impact": ["⬆️", "⬇️", "⬆️"],
            }
        )

        st.dataframe(impact_data, use_container_width=True, hide_index=True)

    st.subheader("Coherence by Level")

    # Chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    levels = ["Item-Store", "Category-Store", "State", "Total"]
    bottom_up = [0.10, 0.02, 0.01, 0.001]
    top_down = [0.15, 0.08, 0.02, 0.005]
    mint = [0.05, 0.01, 0.005, 0.002]
    ols = [0.08, 0.03, 0.01, 0.003]

    x = np.arange(len(levels))
    width = 0.2

    ax.bar(x - 1.5 * width, bottom_up, width, label="Bottom-Up")
    ax.bar(x - 0.5 * width, top_down, width, label="Top-Down")
    ax.bar(x + 0.5 * width, mint, width, label="MinT")
    ax.bar(x + 1.5 * width, ols, width, label="OLS")

    ax.set_xlabel("Hierarchy Level")
    ax.set_ylabel("Coherence Error")
    ax.set_title("Coherence Error by Reconciliation Method")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    st.pyplot(fig)

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 12px; margin-top: 30px;">
    <p>Hierarchical Demand Forecasting System v0.1.0</p>
    <p>Built with Streamlit • Data from M5 Dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)
