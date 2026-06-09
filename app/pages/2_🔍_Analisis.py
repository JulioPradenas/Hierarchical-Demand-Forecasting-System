"""
Página 2: 🔍 Análisis de Datos (Explore Data / EDA)
Tres tabs: Distribución, Correlación, Insights de Jerarquía
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.config import COLORS, HIERARCHY_LEVELS
from app.utils import load_demo_data, render_gradient_header

# Configurar página
st.set_page_config(
    page_title="Análisis - Portfolio",
    page_icon="🔍",
    layout="wide",
)

# Header
render_gradient_header(
    title="Análisis Exploratorio de Datos",
    subtitle="Distribuciones, correlaciones e insights de la jerarquía",
)

# Cargar datos
forecast_df, hierarchy_df = load_demo_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Distribución", "🔗 Correlación", "🌳 Jerarquía"])

# ========== TAB 1: DISTRIBUCIÓN ==========
with tab1:
    st.markdown("Visualiza la distribución de la demanda histórica por producto-tienda")

    # Filtro por nivel jerárquico
    col1, col2 = st.columns([2, 1])
    with col2:
        selected_level = st.selectbox(
            "Nivel Jerárquico:",
            HIERARCHY_LEVELS,
            index=0,
        )

    # Estadísticas de la demanda
    demand = forecast_df[forecast_df["actual"].notna()]["actual"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Media", f"{demand.mean():.1f}")
    with col2:
        st.metric("Mediana", f"{demand.median():.1f}")
    with col3:
        st.metric("Desv. Est.", f"{demand.std():.1f}")
    with col4:
        st.metric("Mín", f"{demand.min():.0f}")
    with col5:
        st.metric("Máx", f"{demand.max():.0f}")

    st.markdown("")

    # Gráfico de distribución
    fig = px.histogram(
        forecast_df[forecast_df["actual"].notna()],
        x="actual",
        nbins=30,
        title="Distribución de Demanda Histórica",
        labels={"actual": "Demanda (unidades)"},
        color_discrete_sequence=[COLORS["primary_start"]],
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Demanda (unidades)")
    fig.update_yaxes(title_text="Frecuencia")

    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 2: CORRELACIÓN ==========
with tab2:
    st.markdown("Matriz de correlación entre features principales y demanda")

    # Generar correlation matrix ficticia
    features = [
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14",
        "fourier_sin_365", "fourier_cos_365",
        "event_distance", "dayofweek"
    ]

    # Generar datos de correlación (demo)
    np.random.seed(42)
    corr_matrix = np.random.uniform(-0.3, 0.8, size=(len(features), len(features)))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Symmetric
    np.fill_diagonal(corr_matrix, 1.0)

    corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)

    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlación"),
    ))
    fig.update_layout(
        title="Matriz de Correlación de Features",
        template="plotly_white",
        height=500,
        xaxis_title="Features",
        yaxis_title="Features",
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 3: JERARQUÍA ==========
with tab3:
    st.markdown("Métricas agregadas por nivel jerárquico")

    # Tabla de niveles
    hierarchy_stats = pd.DataFrame({
        "Nivel": ["Producto-Tienda", "Categoría-Tienda", "Tienda", "Total"],
        "Cantidad de Series": [9, 3, 3, 1],
        "Demanda Promedio": [52.1, 156.3, 468.9, 468.9],
        "Volatilidad": [18.2, 15.6, 12.1, 8.3],
    })

    st.dataframe(hierarchy_stats, use_container_width=True, hide_index=True)

    st.markdown("")

    # Gráfico de cantidad de series por nivel
    fig = px.bar(
        hierarchy_stats,
        x="Nivel",
        y="Cantidad de Series",
        title="Series Temporales por Nivel Jerárquico",
        color_discrete_sequence=[COLORS["primary_start"]],
    )
    fig.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
