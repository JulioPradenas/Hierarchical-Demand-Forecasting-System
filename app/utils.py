from __future__ import annotations

import html
import numpy as np
import pandas as pd
import streamlit as st

from app.config import COLORS, DEMO_PRODUCTS, DEMO_STORES


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load demo forecast and hierarchy data with 100 days historical + 28 days future."""
    rng = np.random.default_rng(42)

    # Generate dates: 100 days historical + 28 days future
    historical_dates = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=100, freq="D")
    future_dates = pd.date_range(start=pd.Timestamp("2025-01-01"), periods=28, freq="D")
    all_dates = historical_dates.append(future_dates)

    # Create combinations of products and stores
    products = DEMO_PRODUCTS
    stores = DEMO_STORES

    # Generate forecast data
    records = []
    base_values = rng.normal(100, 15, len(products) * len(stores))
    idx = 0

    for product in products:
        for store in stores:
            base = base_values[idx]
            idx += 1

            for date in all_dates:
                is_historical = date <= pd.Timestamp("2024-12-31")

                if is_historical:
                    actual = base + rng.normal(0, 5)
                    forecast = actual + rng.normal(0, 2)
                else:
                    actual = np.nan
                    forecast = base + rng.normal(0, 3)

                lower_80 = forecast - rng.uniform(5, 15)
                upper_80 = forecast + rng.uniform(5, 15)
                lower_95 = forecast - rng.uniform(10, 20)
                upper_95 = forecast + rng.uniform(10, 20)

                records.append({
                    "date": date,
                    "item_id": product,
                    "store_id": store,
                    "actual": actual if is_historical else np.nan,
                    "forecast": forecast,
                    "lower_80": lower_80,
                    "upper_80": upper_80,
                    "lower_95": lower_95,
                    "upper_95": upper_95,
                })

    forecast_df = pd.DataFrame(records)

    # Generate hierarchy data
    hierarchy_records = []
    for product in products:
        for store in stores:
            category = ["Lácteos", "Lácteos", "Quesos"][products.index(product)]
            region = ["Centro", "Providencia", "Ñuñoa"][stores.index(store)]

            hierarchy_records.append({
                "item_id": product,
                "category": category,
                "store_id": store,
                "region": region,
                "wrmsse": rng.uniform(0.7, 1.2),
                "coherence_error": rng.uniform(0.01, 0.1),
            })

    hierarchy_df = pd.DataFrame(hierarchy_records)

    return forecast_df, hierarchy_df


def render_gradient_header(title: str, subtitle: str) -> None:
    """Render a gradient header with title and subtitle."""
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    primary_start = COLORS['primary_start']
    primary_end = COLORS['primary_end']
    html_content = f"""
    <div style="
        background: linear-gradient(135deg, {primary_start} 0%, {primary_end} 100%);
        color: white;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <h1 style="
            font-size: 28px;
            font-weight: bold;
            margin: 0 0 10px 0;
        ">{safe_title}</h1>
        <p style="font-size: 16px; margin: 0;">{safe_subtitle}</p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta: str | None, accent_color: str) -> None:
    """Render a KPI card with accent bar and optional delta."""
    color = COLORS.get(accent_color, COLORS["primary_start"])
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_delta = html.escape(delta) if delta else ""
    delta_html = (
        f"<span style='font-size: 12px; color: {color};'>{safe_delta}</span>"
        if delta
        else ""
    )

    bg_light = COLORS['bg_light']
    border_color = COLORS['border']
    text_secondary = COLORS['text_secondary']
    html_content = f"""
    <div style="
        border-left: 4px solid {color};
        border-right: 1px solid {border_color};
        border-top: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
        background: {bg_light};
        border-radius: 4px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    ">
        <p style="
            font-size: 12px;
            color: {text_secondary};
            margin: 0 0 8px 0;
        ">{safe_label}</p>
        <p style="
            font-size: 20px;
            font-weight: bold;
            color: {color};
            margin: 0 0 4px 0;
        ">{safe_value}</p>
        {delta_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_kpi_grid(kpis: list[tuple[str, str, str | None, str]]) -> None:
    """Render KPIs in a 2-column grid layout."""
    cols = st.columns(2)
    for i, (label, value, delta, color_key) in enumerate(kpis):
        with cols[i % 2]:
            render_kpi_card(label, value, delta, color_key)


def render_tech_stack_badges(technologies: list[str]) -> None:
    """Render technology badges with gradient background."""
    primary_start = COLORS['primary_start']
    primary_end = COLORS['primary_end']
    badges_html = ""
    for tech in technologies:
        safe_tech = html.escape(tech)
        badges_html += f"""
        <span style="
            background: linear-gradient(
                135deg,
                {primary_start} 0%,
                {primary_end} 100%
            );
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            display: inline-block;
            margin-right: 8px;
            margin-bottom: 8px;
        ">{safe_tech}</span>
        """

    html_content = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        {badges_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
