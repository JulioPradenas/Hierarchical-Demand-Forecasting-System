# Global configuration: colors, URLs, and shared constants

COLORS = {
    "primary_start": "#667eea",
    "primary_end": "#764ba2",
    "success": "#22c55e",
    "error": "#ef4444",
    "info": "#0ea5e9",
    "bg_light": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "text_primary": "#1f2937",
    "text_secondary": "#6b7280",
    "border": "#e5e7eb",
}

GITHUB_REPO = "https://github.com/JulioPradenas/Hierarchical-Demand-Forecasting-System"
GITHUB_DOCS = f"{GITHUB_REPO}/blob/main/README.md"

DEMO_PRODUCTS = [
    "Leche Descremada 1L",
    "Yogur Natural 500g",
    "Queso Mozzarella 250g",
]

DEMO_STORES = [
    "Supermercado Centro - Santiago",
    "Supermercado Mall - Providencia",
    "Tienda Express - Ñuñoa",
]

KPI_PRODUCTS = 3049
KPI_MASE = 0.89
KPI_IMPROVEMENT = 14.5
KPI_HIERARCHY_LEVELS = 6

DEMO_FEATURES = [
    ("Price", 0.18),
    ("Promotion", 0.15),
    ("Seasonality", 0.12),
    ("Trend", 0.11),
    ("Competitor Price", 0.10),
    ("Stock Level", 0.09),
    ("Day of Week", 0.08),
    ("Temperature", 0.07),
    ("Holiday", 0.06),
    ("Rainfall", 0.05),
    ("Traffic", 0.04),
    ("Social Media", 0.03),
    ("Discount Rate", 0.02),
    ("Product Category", 0.02),
    ("Store Location", 0.01),
]

HIERARCHY_LEVELS = [
    "Producto-Tienda",
    "Categoría-Tienda",
    "Tienda",
    "Total",
]
