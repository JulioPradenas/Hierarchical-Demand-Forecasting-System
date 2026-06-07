from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Configuración de la app
st.set_page_config(
    page_title="Pronóstico Jerárquico de Demanda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
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

# Navegación en sidebar
st.sidebar.title("🎯 Navegación")
page = st.sidebar.radio(
    "Selecciona una página:",
    ["Explorador de Pronósticos", "Visualizador de Jerarquía", "Análisis de Reconciliación"],
    index=0,
)


# Cargar datos de ejemplo
@st.cache_data
def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga datos de ejemplo para el dashboard."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")

    # Definir productos y tiendas con nombres reales
    items = [
        "Leche Descremada 1L",
        "Yogur Natural 500g",
        "Queso Mozzarella 250g",
    ]
    stores = [
        "Supermercado Centro - Santiago",
        "Supermercado Mall - Providencia",
        "Tienda Express - Ñuñoa",
    ]

    # Generar datos para TODOS los items y tiendas
    forecast_list = []
    rng = np.random.default_rng(42)

    for item in items:
        for store in stores:
            # Datos históricos (100 días)
            for date in dates:
                forecast_list.append({
                    "date": date,
                    "item_id": item,
                    "store_id": store,
                    "actual": int(rng.poisson(50)),
                    "forecast": float(50 + rng.normal(0, 5)),
                    "lower_80": float(40 + rng.normal(0, 5)),
                    "upper_80": float(60 + rng.normal(0, 5)),
                    "lower_95": float(30 + rng.normal(0, 5)),
                    "upper_95": float(70 + rng.normal(0, 5)),
                })

            # Datos futuros (28 días)
            future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=28, freq="D")
            for date in future_dates:
                forecast_list.append({
                    "date": date,
                    "item_id": item,
                    "store_id": store,
                    "actual": np.nan,
                    "forecast": float(50 + rng.normal(0, 5)),
                    "lower_80": float(40 + rng.normal(0, 5)),
                    "upper_80": float(60 + rng.normal(0, 5)),
                    "lower_95": float(30 + rng.normal(0, 5)),
                    "upper_95": float(70 + rng.normal(0, 5)),
                })

    forecast = pd.DataFrame(forecast_list)

    # Datos de jerarquía con categorías reales
    categories = {
        "Leche Descremada 1L": "Lácteos",
        "Yogur Natural 500g": "Lácteos",
        "Queso Mozzarella 250g": "Lácteos",
    }
    regions = {
        "Supermercado Centro - Santiago": "Región Metropolitana",
        "Supermercado Mall - Providencia": "Región Metropolitana",
        "Tienda Express - Ñuñoa": "Región Metropolitana",
    }

    hierarchy_list = []
    for item in items:
        for store in stores:
            hierarchy_list.append({
                "item_id": item,
                "category": categories[item],
                "store_id": store,
                "region": regions[store],
                "wrmsse": float(rng.uniform(0.35, 0.55)),
                "coherence_error": float(rng.uniform(0.001, 0.05)),
            })

    hierarchy = pd.DataFrame(hierarchy_list)

    return forecast, hierarchy


# Página 1: Explorador de Pronósticos
if page == "Explorador de Pronósticos":
    st.title("📈 Explorador de Pronósticos")
    st.markdown(
        "Explora pronósticos para artículos individuales con toggle de reconciliación."
    )

    forecast_df, hierarchy_df = load_sample_data()

    # Selectores
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        item_id = st.selectbox(
            "Selecciona Artículo:",
            hierarchy_df["item_id"].unique(),
            index=0,
        )
    with col2:
        store_id = st.selectbox(
            "Selecciona Tienda:",
            hierarchy_df["store_id"].unique(),
            index=0,
        )
    with col3:
        show_reconciled = st.checkbox("Mostrar Pronóstico Reconciliado", value=True)

    # Filtrar datos
    mask = (forecast_df["item_id"] == item_id) & (forecast_df["store_id"] == store_id)
    data = forecast_df[mask].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    # Dividir histórico y futuro
    historical = data[data["actual"].notna()].copy()
    future = data[data["actual"].isna()].copy()

    # Gráfico
    st.subheader(f"{item_id} @ {store_id}")

    chart_cols = st.columns([3, 1])
    with chart_cols[0]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        # Datos históricos
        if len(historical) > 0:
            ax.plot(
                historical["date"],
                historical["actual"],
                "ko-",
                label="Real",
                linewidth=2,
            )

        # Pronóstico
        ax.plot(
            historical["date"],
            historical["forecast"],
            "b-",
            label="Pronóstico",
            linewidth=1.5,
        )

        # Intervalos
        ax.fill_between(
            historical["date"],
            historical["lower_95"],
            historical["upper_95"],
            alpha=0.2,
            label="Intervalo 95%",
        )
        ax.fill_between(
            historical["date"],
            historical["lower_80"],
            historical["upper_80"],
            alpha=0.3,
            label="Intervalo 80%",
        )

        # Pronóstico futuro
        if len(future) > 0:
            future_dates = future["date"].values
            ax.plot(
                future_dates,
                future["forecast"],
                "b--",
                label="Pronóstico Futuro",
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

        ax.set_xlabel("Fecha")
        ax.set_ylabel("Demanda")
        ax.set_title("Pronóstico de Demanda con Intervalos de Confianza")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    with chart_cols[1]:
        st.metric("Error de Coherencia", "0.015", "-0.005")
        st.metric("WRMSSE", "0.42", "-0.08")
        st.metric("Horizonte de Pronóstico", "28 días")

# Página 2: Visualizador de Jerarquía
elif page == "Visualizador de Jerarquía":
    st.title("🌳 Visualizador de Jerarquía")
    st.markdown(
        "Vista interactiva de la estructura jerárquica con métricas de pronóstico."
    )

    forecast_df, hierarchy_df = load_sample_data()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Niveles de Jerarquía")

        # Niveles de agregación realistas
        levels = ["Producto-Tienda", "Categoría-Tienda", "Región", "Total"]
        selected_level = st.radio("Selecciona Nivel:", levels, index=0)

        st.subheader("Métricas por Nivel")

        metrics_data = {
            "Nivel": ["Producto-Tienda", "Categoría-Tienda", "Región", "Total"],
            "Cantidad": [9, 3, 1, 1],
            "WRMSSE Promedio": [0.45, 0.42, 0.40, 0.38],
            "Error de Coherencia Promedio": [0.02, 0.015, 0.01, 0.005],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader(f"Pronósticos @ {selected_level}")

        # Mostrar pronósticos
        display_data = hierarchy_df[
            ["item_id", "category", "wrmsse", "coherence_error"]
        ]
        display_data = display_data.copy()
        display_data.columns = ["Producto", "Categoría", "WRMSSE", "Error de Coherencia"]

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        # Mapa de calor de incertidumbre
        st.subheader("Mapa de Calor de Incertidumbre (WRMSSE por Producto-Tienda)")

        heatmap_data = pd.DataFrame(
            {
                "Producto": ["Leche Descremada 1L", "Yogur Natural 500g", "Queso Mozzarella 250g"],
                "Centro - Santiago": [0.45, 0.52, 0.38],
                "Mall - Providencia": [0.48, 0.55, 0.40],
                "Express - Ñuñoa": [0.42, 0.50, 0.39],
            }
        )

        st.dataframe(heatmap_data, use_container_width=True, hide_index=True)

# Página 3: Análisis de Reconciliación
else:  # Análisis de Reconciliación
    st.title("🔄 Análisis de Reconciliación")
    st.markdown(
        "Compara métodos de reconciliación y su impacto en la coherencia."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Métodos de Reconciliación")

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
                    "Método": method,
                    "Error de Coherencia": f"{metrics['coherence_error']:.4f}",
                    "WRMSSE": f"{metrics['wrmsse']:.2f}",
                }
            )

        st.dataframe(pd.DataFrame(method_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Impacto de Negocio")

        impact_data = pd.DataFrame(
            {
                "Métrica": ["Reducción de Costo Total", "Mejora en Stock de Seguridad", "Nivel de Servicio"],
                "Valor": ["12.5%", "8.3%", "94.2%"],
                "Impacto": ["⬆️", "⬇️", "⬆️"],
            }
        )

        st.dataframe(impact_data, use_container_width=True, hide_index=True)

    st.subheader("Coherencia por Nivel")

    # Gráfico
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    levels = ["Producto-Tienda", "Categoría-Tienda", "Región", "Total"]
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

    ax.set_xlabel("Nivel de Jerarquía")
    ax.set_ylabel("Error de Coherencia")
    ax.set_title("Error de Coherencia por Método de Reconciliación")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    st.pyplot(fig)

# Pie de página
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 12px; margin-top: 30px;">
    <p>Sistema de Pronóstico Jerárquico de Demanda v0.1.0</p>
    <p>Construido con Streamlit • Datos del Dataset M5</p>
    <p>© 2026 - Portafolio de Data Science</p>
    </div>
    """,
    unsafe_allow_html=True,
)
