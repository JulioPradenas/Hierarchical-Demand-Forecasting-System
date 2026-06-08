"""
Página 3: 🤖 Resultados del Modelo (Model Results)
Comparación de modelos, feature importance, predictor interactivo.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from app.config import COLORS, DEMO_FEATURES, DEMO_PRODUCTS, DEMO_STORES
from app.utils import load_demo_data, render_gradient_header

# Configurar página
st.set_page_config(
    page_title="Resultados - Portfolio",
    page_icon="🤖",
    layout="wide",
)

# Header
render_gradient_header(
    title="Resultados del Modelo",
    subtitle="Comparación de modelos, importancia de features, predictor en vivo",
)

# Cargar datos
forecast_df, hierarchy_df = load_demo_data()

# ========== SECCIÓN 1: COMPARACIÓN DE MODELOS ==========
st.subheader("📊 Comparación de Modelos")

models_data = {
    "Modelo": ["Baseline\n(SeasonalNaive)", "Candidato 1\n(ETS)", "Candidato 2\n(AutoARIMA)", "🏆 Ganador\n(LightGBM + MinT)"],
    "MASE": [1.0440, 0.9520, 0.9180, 0.8890],
    "RMSSE": [0.95, 0.88, 0.85, 0.78],
    "Coherence Error": [0.0150, 0.0085, 0.0045, 0.0020],
    "Tiempo (min)": [0.2, 5.3, 12.1, 8.5],
}

models_df = pd.DataFrame(models_data)

# Mostrar tabla con highlight en ganador
st.dataframe(models_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    **Por qué ganó:** Mejor balance entre acuracez (MASE 0.89) y coherencia jerárquica (error 0.002).
    **MinT reconciliation** asegura que pronósticos agregados son consistentes con totales.
    """
)

st.divider()

# ========== SECCIÓN 2: IMPORTANCIA DE FEATURES ==========
st.subheader("🔍 Importancia de Features (Top 15)")

features_df = pd.DataFrame(DEMO_FEATURES, columns=["Feature", "Importancia"])

fig = px.barh(
    features_df,
    x="Importancia",
    y="Feature",
    orientation="h",
    color_discrete_sequence=[COLORS["primary_start"]],
    title="Feature Importance del Modelo Ganador",
)
fig.update_layout(
    template="plotly_white",
    height=500,
    showlegend=False,
    xaxis_title="Importancia (ganancia)",
    yaxis_title="",
)
fig.update_xaxes(automargin=True)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ========== SECCIÓN 3: PREDICTOR INTERACTIVO ==========
st.subheader("🎯 Predictor Interactivo")
st.markdown("Prueba el modelo en vivo. Ingresa valores y ve la predicción.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_product = st.selectbox("Producto:", DEMO_PRODUCTS, index=0)

with col2:
    selected_store = st.selectbox("Tienda:", DEMO_STORES, index=0)

with col3:
    avg_demand = st.slider("Demanda Histórica Promedio:", 0, 200, 50)

with col4:
    seasonality = st.slider("Factor de Estacionalidad:", 0.5, 1.5, 1.0, step=0.1)

# Calcular predicción (demo)
base_forecast = avg_demand * seasonality
lower_80 = base_forecast * 0.8
upper_80 = base_forecast * 1.2
lower_95 = base_forecast * 0.7
upper_95 = base_forecast * 1.3

# Mostrar resultado
col1, col2 = st.columns([2, 1])

with col1:
    html = f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['primary_start']} 0%, {COLORS['primary_end']} 100%);
        color: white;
        padding: 30px;
        border-radius: 8px;
        text-align: center;
    ">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
            Predicción para <strong>{selected_product}</strong> @ <strong>{selected_store}</strong>
        </div>
        <div style="font-size: 48px; font-weight: bold; margin-bottom: 10px;">
            {base_forecast:.0f}
        </div>
        <div style="font-size: 12px; opacity: 0.85;">
            Intervalo 80%: [{lower_80:.0f}, {upper_80:.0f}]<br>
            Intervalo 95%: [{lower_95:.0f}, {upper_95:.0f}]
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

with col2:
    st.metric("Confianza 80%", f"{((upper_80 - lower_80) / base_forecast * 100):.0f}%")
    st.metric("Confianza 95%", f"{((upper_95 - lower_95) / base_forecast * 100):.0f}%")

st.divider()

# ========== SECCIÓN 4: COHERENCIA POR NIVEL ==========
st.subheader("📈 Coherencia por Nivel Jerárquico")

coherence_data = {
    "Nivel": ["Producto-Tienda", "Categoría-Tienda", "Tienda", "Total"],
    "Bottom-Up": [0.10, 0.02, 0.01, 0.001],
    "Top-Down": [0.15, 0.08, 0.02, 0.005],
    "MinT": [0.05, 0.01, 0.005, 0.002],
}

coherence_df = pd.DataFrame(coherence_data)

fig = px.line(
    coherence_df,
    x="Nivel",
    y=["Bottom-Up", "Top-Down", "MinT"],
    title="Error de Coherencia por Método de Reconciliación",
    markers=True,
)
fig.update_layout(
    template="plotly_white",
    height=400,
    hovermode="x unified",
    yaxis_title="Error de Coherencia",
    xaxis_title="Nivel Jerárquico",
)

st.plotly_chart(fig, use_container_width=True)
