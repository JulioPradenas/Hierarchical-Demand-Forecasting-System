"""
Página 4: ⚙️ Cómo Construí Esto (How I Built This)
Arquitectura, timeline de desarrollo, decisiones clave.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from app.config import COLORS, GITHUB_REPO
from app.utils import render_gradient_header

# Configurar página
st.set_page_config(
    page_title="Proceso - Portfolio",
    page_icon="⚙️",
    layout="wide",
)

# Header
render_gradient_header(
    title="Cómo Construí Este Sistema",
    subtitle="Arquitectura, decisiones clave, lecciones aprendidas",
)

# ========== SECCIÓN 1: ARQUITECTURA ==========
st.subheader("🏗️ Arquitectura del Sistema")

arch_html = f"""
<div style="
    background: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 20px;
    font-family: monospace;
    font-size: 12px;
    overflow-x: auto;
">
<pre>
Datos M5
   ↓
[DuckDB] Carga & Procesamiento
   ↓
[Feature Engineering]
   ├─ Temporal: lags, rolling stats, Fourier
   ├─ Calendar: 22 eventos codificados
   └─ Hierarchical: agregaciones
   ↓
[Modelo Global] LightGBM
   ├─ Entrada: 50+ features
   ├─ Salida: Pronóstico puntual + intervalos
   └─ Time-Series Cross-Validation: 3 folds
   ↓
[Reconciliación] MinT
   ├─ Método: mint_shrink
   ├─ Asegura: Coherencia jerárquica
   └─ Salida: Pronósticos consistentes
   ↓
[API REST] FastAPI
   ├─ Endpoints: /predict, /models, /hierarchy
   └─ Documentación: /docs
   ↓
[Dashboard] Streamlit (esta app)
</pre>
</div>
"""
st.markdown(arch_html, unsafe_allow_html=True)

st.markdown("")

# ========== SECCIÓN 2: TIMELINE DE DESARROLLO ==========
st.subheader("📅 Timeline de Desarrollo")

timeline_items = [
    {
        "semana": "Semana 1",
        "titulo": "🔍 Data Exploration & Preprocessing",
        "items": [
            "Cargué dataset M5 (3,049 productos, 10 tiendas, 3 estados, 1,913 días)",
            "Analizé estructura jerárquica: 6 niveles",
            "Exploré patrones de seasonalidad y eventos M5",
        ],
    },
    {
        "semana": "Semana 2",
        "titulo": "⚡ Feature Engineering",
        "items": [
            "Temporal features: lags [7,14,28,35...365], rolling stats, Fourier terms",
            "Calendar features: 22 eventos M5 codificados como distancia",
            "Hierarchical features: agregaciones bottom-up",
        ],
    },
    {
        "semana": "Semana 3",
        "titulo": "🤖 Modeling & Reconciliation",
        "items": [
            "Baseline: SeasonalNaive (MASE 1.04)",
            "Candidatos: ETS, AutoARIMA, LightGBM (tuned con Optuna)",
            "Ganador: LightGBM + MinT (MASE 0.89, coherence error 0.002)",
        ],
    },
]

for item in timeline_items:
    with st.expander(f"**{item['semana']}** — {item['titulo']}", expanded=False):
        for subitem in item["items"]:
            st.markdown(f"- {subitem}")

st.divider()

# ========== SECCIÓN 3: DECISIONES CLAVE ==========
st.subheader("💡 Decisiones Clave")

decisions = [
    {
        "pregunta": "¿Por qué LightGBM?",
        "respuesta": (
            "**Velocidad:** Entrena 10x más rápido que XGBoost en datasets grandes.\n"
            "**Precisión:** Captura no-linearidades y feature interactions mejor que estadísticos puros.\n"
            "**Escalabilidad:** Maneja 50+ features sin problemas. Usado en competencias Kaggle de forecasting."
        ),
    },
    {
        "pregunta": "¿Por qué MinT reconciliation?",
        "respuesta": (
            "**Coherencia Jerárquica:** Asegura que pronósticos agregados = suma de base.\n"
            "**Matemático:** Mínimos cuadrados + shrinkage → óptimo under MSE.\n"
            "**Alternativas descartadas:** Bottom-Up (pierde señal agregada), Top-Down (complejidad asignación)."
        ),
    },
    {
        "pregunta": "¿Por qué 6 niveles de jerarquía?",
        "respuesta": (
            "**Estructura real M5:** Total → Estado → Tienda → Categoría → Departamento → Item-Tienda.\n"
            "**Ventaja:** Reconsiliación en cada nivel captura dependencies locales.\n"
            "**Reto:** Aumenta costo computacional de cuadrático a cúbico."
        ),
    },
    {
        "pregunta": "¿Cómo manejaste el desbalance temporal?",
        "respuesta": (
            "**TimeSeriesCV:** Expanding window, 3 folds, horizon=28 días.\n"
            "**No data leakage:** Futuro nunca entra en train antes que aparezca en test.\n"
            "**Validación:** WRMSSE calculado per-fold, promediado."
        ),
    },
]

for decision in decisions:
    with st.expander(f"**{decision['pregunta']}**", expanded=False):
        st.markdown(decision["respuesta"])

st.divider()

# ========== SECCIÓN 4: LINKS Y REFERENCIAS ==========
st.subheader("🔗 Links y Recursos")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"[📂 Código en GitHub]({GITHUB_REPO})")

with col2:
    st.markdown(f"[📖 README & Docs]({GITHUB_REPO}/blob/main/README.md)")

with col3:
    st.markdown(
        f"[📄 Paper MinT (Wickramasuriya et al., 2019)]"
        f"(https://robjhyndman.com/papers/mintheir.pdf)",
        unsafe_allow_html=True,
    )

st.markdown("")
st.markdown("---")

# Footer
st.markdown(
    f"""
    <div style="text-align: center; color: {COLORS['text_secondary']}; font-size: 12px; margin-top: 20px;">
    <p>Construido con ❤️ usando Python, LightGBM, MinT, Streamlit</p>
    <p><strong>© 2026 - Hierarchical Demand Forecasting System</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)
