"""
data_preprocessing.py
---------------------
Load the raw Telco data, clean it, engineer business-relevant features, and
write a tidy, analysis-ready dataset to data/processed/.

Run directly:  python src/data_preprocessing.py
"""

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # allow running as a plain script
    import config


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA)
    print(f"[load]   raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# --------------------------------------------------------------------------- #
# 2. Clean
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is read as text because new customers (tenure = 0) have a
    # blank value. Coerce to numeric and backfill those blanks with 0 — they
    # have not been billed a full cycle yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"[clean]  fixed {n_blank} blank TotalCharges (new, tenure=0 customers)")

    # SeniorCitizen ships as 0/1 — relabel to Yes/No for readable charts.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Normalise the many "No internet service" / "No phone service" labels to a
    # plain "No" so categories don't fragment during encoding and plotting.
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df[service_cols] = df[service_cols].replace(
        {"No internet service": "No", "No phone service": "No"}
    )

    # Drop the identifier — useless for modelling, kept out to avoid leakage.
    df = df.drop(columns=["customerID"])

    # Trim stray whitespace in object columns.
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    print(f"[clean]  duplicates removed: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets — turns a continuous field into an intuitive lifecycle stage.
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-5 yr", "5+ yr"],
    )

    # Count of value-added services each customer subscribes to (engagement proxy).
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumAddOnServices"] = (df[addon_cols] == "Yes").sum(axis=1)

    # Average spend per tenure month — separates high-value from low-value churners.
    df["AvgChargesPerMonth"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Flags that EDA repeatedly shows are the sharpest churn signals.
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["HasFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)

    print(f"[feateng] engineered features -> new shape: {df.shape[1]} cols")
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_processed_dataset(save: bool = True) -> pd.DataFrame:
    df = engineer(clean(load_raw()))
    if save:
        df.to_csv(config.PROCESSED_DATA, index=False)
        print(f"[save]   processed dataset -> {config.PROCESSED_DATA}")
    churn_rate = (df[config.TARGET] == "Yes").mean()
    print(f"[done]   final shape: {df.shape[0]:,} x {df.shape[1]} | "
          f"churn rate: {churn_rate:.1%}")
    return df


if __name__ == "__main__":
    build_processed_dataset()
