import streamlit as st

from app.config import GITHUB_REPO
from app.utils import (
    render_gradient_header,
    render_kpi_grid,
    render_tech_stack_badges,
)

# Set page config
st.set_page_config(
    page_title="Resumen - Portfolio",
    page_icon="🏠",
    layout="wide",
)

# Render gradient header
render_gradient_header(
    title="🏠 Resumen del Proyecto",
    subtitle="Sistema Jerárquico de Pronóstico de Demanda",
)

# Render KPI grid
kpis = [
    ("Productos", "3,049", None, "primary_start"),
    ("MASE", "0.89", None, "success"),
    ("Mejora vs Baseline", "↓14.5%", None, "error"),
    ("Niveles Jerárquicos", "6", None, "info"),
]
render_kpi_grid(kpis)

# Divider
st.divider()

# Tech stack section
st.subheading("🛠️ Stack Tecnológico")
render_tech_stack_badges(
    [
        "Python",
        "LightGBM",
        "MinT Reconciliation",
        "Streamlit",
        "Plotly",
        "DuckDB",
    ]
)

# CTA text
st.markdown(
    """
    Este proyecto implementa un sistema de pronóstico de demanda jerárquico
    que reconcilia predicciones en múltiples niveles de agregación usando
    técnicas avanzadas de ML y procedimientos de coherencia.

    Explora el código, la documentación y los detalles técnicos en el repositorio.
    """
)

# GitHub link button
st.link_button("📂 Ver en GitHub", GITHUB_REPO)

# Footer
st.markdown(
    """
    <hr style="margin: 40px 0; border: none; border-top: 1px solid #e5e7eb;">
    <div style="
        text-align: center;
        padding: 20px;
        font-size: 12px;
        color: #6b7280;
    ">
        <p>
            Desarrollado por
            <a href="https://github.com/JulioPradenas" style="color: #667eea; text-decoration: none;">
                Julio Pradenas
            </a> |
            <a href="https://github.com/JulioPradenas/Hierarchical-Demand-Forecasting-System"
               style="color: #667eea; text-decoration: none;">
                GitHub
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
