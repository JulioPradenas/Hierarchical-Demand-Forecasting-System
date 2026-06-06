# Hierarchical Demand Forecasting System

Sistema de pronóstico de demanda jerárquico sobre el dataset M5 de Walmart — modelo global LightGBM, reconciliación MinT en 6 niveles de jerarquía, intervalos de predicción calibrados y API REST lista para producción.

## Descripción

En retail y e-commerce, los pronósticos se generan a múltiples niveles: el CFO quiere el total nacional, el director de supply chain quiere por región, el jefe de bodega quiere por ciudad. Si cada nivel se pronostica de forma independiente, los números son **incoherentes** (la suma de las regiones no coincide con el total nacional).

Este proyecto resuelve ese problema con **reconciliación jerárquica MinT (Minimum Trace)**, garantizando coherencia matemática entre los 6 niveles del dataset M5 de Walmart.

**Dataset:** [M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — 3.049 productos × 10 tiendas × 3 estados de EE.UU., benchmark estándar de la industria.

## Jerarquía

```
Total Nacional
├── Estado (CA, TX, WI)
│   └── Tienda (CA_1 ... WI_3)
│       └── Categoría (FOODS, HOBBIES, HOUSEHOLD)
│           └── Departamento (FOODS_1, FOODS_2 ...)
│               └── Ítem-Tienda (3.049 series base)
```

## Stack Técnico

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.11, uv |
| Datos | pandas, polars, pyarrow, DuckDB |
| Modelos | LightGBM, statsforecast (ETS/AutoARIMA) |
| Reconciliación | hierarchicalforecast (MinT, BU, TD, OLS) |
| Tuning | Optuna |
| Explainability | SHAP |
| Tracking | MLflow |
| API | FastAPI |
| Dashboard | Streamlit |
| Calidad | pytest >85% cobertura, ruff, mypy, pre-commit |
| Infra | Docker, GitHub Actions |

## Estructura

```
src/demand_forecast/
├── config/          # Pydantic Settings
├── data/            # M5DataLoader, ForecastHierarchy, DuckDB SQL features
├── features/        # FeatureBuilder ABC, Temporal, Calendar, Hierarchical
├── models/          # LGBMGlobalForecaster, ETS, SeasonalNaive, Ensemble
├── reconciliation/  # MinTReconciler, BU, TD, OLS
├── evaluation/      # WRMSSE, MASE, CRPS, TimeSeriesCV, business metrics
└── pipelines/       # Training & inference pipelines
```

## Instalación

```bash
# Requiere uv (https://docs.astral.sh/uv/)
uv sync
```

## Uso

```bash
# Tests
make test

# Lint + typecheck + tests
make check

# Descargar dataset M5 (requiere cuenta Kaggle)
export KAGGLE_USERNAME=tu_usuario
export KAGGLE_KEY=tu_api_key
uv run python -c "
from pathlib import Path
from demand_forecast.data.loader import M5DataLoader
M5DataLoader(Path('data/raw')).download()
"
```

## Fases del Proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Setup, CI/CD, scaffolding | Completo |
| 2 | M5DataLoader, ForecastHierarchy, matriz S | Completo |
| 3 | Feature engineering (temporal, calendar, hierarchical, SQL) | Completo |
| 4 | Baselines: SeasonalNaive, ETS, AutoARIMA | En desarrollo |
| 5 | LightGBM global + Optuna + SHAP | Pendiente |
| 6 | Reconciliación MinT vs BU/TD/OLS | Pendiente |
| 7 | Forecasting probabilístico (CRPS, conformal prediction) | Pendiente |
| 8 | Evaluación de negocio (costo asimétrico, safety stock) | Pendiente |
| 9 | API FastAPI con coherence check | Pendiente |
| 10 | Dashboard Streamlit + deploy | Pendiente |

## Métricas Objetivo

| Métrica | Descripción |
|---------|-------------|
| WRMSSE | Weighted Root Mean Squared Scaled Error (estándar M5) |
| MASE | Mean Absolute Scaled Error por nivel jerárquico |
| CRPS | Continuous Ranked Probability Score (calidad de intervalos) |
| Coverage | % de valores reales dentro del intervalo de predicción |

## Diferenciadores

- **Reconciliación MinT** — el estándar de la industria, casi nunca visto en portfolios
- **6 niveles simultáneos** — arquitectura de modelos coherente, no solo un modelo aislado
- **CRPS como métrica probabilística** — más allá del forecast puntual
- **Benchmark M5** — dataset de competencia Kaggle de referencia mundial
- **Stack productivo completo** — MLflow, FastAPI, Streamlit, Docker, GitHub Actions
