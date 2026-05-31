"""
config.py
---------
Central configuration: file paths, business assumptions, and the shared
visual theme used across every figure so the project looks consistent.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: writes PNGs, no GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, so scripts run from anywhere)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "telco_churn_clean.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
METRICS_FILE = ROOT / "reports" / "model_metrics.csv"

for _d in (PROCESSED_DATA.parent, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Business assumptions (used for the revenue / ROI analysis).
# These are transparent, documented levers a stakeholder could challenge.
# --------------------------------------------------------------------------- #
AVG_RETENTION_OFFER_COST = 8.0      # $/customer/month cost of a retention offer
RETENTION_CAMPAIGN_SUCCESS = 0.30   # share of targeted churners we actually save
EXPECTED_LIFETIME_MONTHS = 12       # horizon used to value a saved customer

RANDOM_STATE = 42
TARGET = "Churn"

# --------------------------------------------------------------------------- #
# Shared visual theme — a single place that makes every chart look polished.
# --------------------------------------------------------------------------- #
NAVY = "#1f2a44"
TEAL = "#2a9d8f"
CORAL = "#e76f51"
AMBER = "#e9c46a"
SLATE = "#8d99ae"
CHURN_PALETTE = {"No": TEAL, "Yes": CORAL}


def set_theme() -> None:
    """Apply the project-wide matplotlib / seaborn styling."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelcolor": NAVY,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.color": "#e6e8ee",
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
