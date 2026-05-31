"""
eda.py
------
Generate the exploratory-data-analysis figure set. Every chart is saved to
reports/figures/ as a polished PNG that can be dropped straight into the
README, a slide deck, or a report.

Run directly:  python src/eda.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _load() -> pd.DataFrame:
    if not config.PROCESSED_DATA.exists():
        from data_preprocessing import build_processed_dataset
        return build_processed_dataset()
    return pd.read_csv(config.PROCESSED_DATA)


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig]  {path.name}")


def _annotate_bars(ax, fmt="{:.0f}%", scale=1.0, pad=3):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h * scale),
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10,
                    color=config.NAVY, xytext=(0, pad),
                    textcoords="offset points")


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #
def fig_churn_overview(df):
    """Headline: how big is the problem and what revenue does it touch?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts = df[config.TARGET].value_counts()
    rate = counts.get("Yes", 0) / counts.sum()
    axes[0].pie(counts, labels=["Retained", "Churned"], autopct="%1.1f%%",
                startangle=90, colors=[config.TEAL, config.CORAL],
                wedgeprops=dict(width=0.42, edgecolor="white"),
                textprops=dict(color=config.NAVY, fontsize=13))
    axes[0].set_title("Customer Churn Rate")
    axes[0].text(0, 0, f"{rate:.0%}\nchurn", ha="center", va="center",
                 fontsize=20, fontweight="bold", color=config.CORAL)

    rev = df.groupby(config.TARGET)["MonthlyCharges"].sum()
    bars = axes[1].bar(["Retained", "Churned"], rev.values,
                       color=[config.TEAL, config.CORAL])
    axes[1].set_title("Monthly Recurring Revenue by Status")
    axes[1].set_ylabel("Total Monthly Charges ($)")
    for b, v in zip(bars, rev.values):
        axes[1].annotate(f"${v/1000:,.1f}k",
                         (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontweight="bold")
    fig.suptitle("Churn at a Glance", fontsize=18, fontweight="bold",
                 color=config.NAVY)
    _save(fig, "01_churn_overview.png")


def _churn_rate_by(df, col, order=None):
    g = df.groupby(col, observed=True)[config.TARGET].apply(
        lambda s: (s == "Yes").mean())
    if order is not None:
        g = g.reindex(order)
    return g


def fig_churn_drivers(df):
    """Four categorical drivers, each shown as a churn-rate bar chart."""
    specs = [
        ("Contract", ["Month-to-month", "One year", "Two year"], "Contract Type"),
        ("InternetService", ["Fiber optic", "DSL", "No"], "Internet Service"),
        ("PaymentMethod", None, "Payment Method"),
        ("TenureGroup", ["0-1 yr", "1-2 yr", "2-4 yr", "4-5 yr", "5+ yr"],
         "Customer Tenure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    overall = (df[config.TARGET] == "Yes").mean()

    for ax, (col, order, title) in zip(axes.ravel(), specs):
        g = _churn_rate_by(df, col, order).sort_values(ascending=False) \
            if order is None else _churn_rate_by(df, col, order)
        colors = [config.CORAL if v >= overall else config.TEAL for v in g.values]
        bars = ax.bar(range(len(g)), g.values, color=colors)
        ax.axhline(overall, ls="--", lw=1.5, color=config.NAVY, alpha=0.7)
        ax.text(len(g) - 0.5, overall + 0.01, f"avg {overall:.0%}",
                ha="right", color=config.NAVY, fontsize=10)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([str(x).replace(" (automatic)", "") for x in g.index],
                           rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(g.values) * 1.25)
        ax.set_title(title)
        for b, v in zip(bars, g.values):
            ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontweight="bold", fontsize=11)
    fig.suptitle("What Drives Churn? — Churn Rate by Customer Segment",
                 fontsize=18, fontweight="bold", color=config.NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "02_churn_drivers.png")


def fig_tenure_charges(df):
    """How tenure and monthly spend separate churners from loyal customers."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for label, sub in df.groupby(config.TARGET):
        axes[0].hist(sub["tenure"], bins=30, alpha=0.65,
                     label="Churned" if label == "Yes" else "Retained",
                     color=config.CHURN_PALETTE[label])
    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Customers")
    axes[0].legend()

    data = [df.loc[df[config.TARGET] == k, "MonthlyCharges"] for k in ["No", "Yes"]]
    parts = axes[1].violinplot(data, showmeans=True)
    for pc, c in zip(parts["bodies"], [config.TEAL, config.CORAL]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        parts[key].set_color(config.NAVY)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Retained", "Churned"])
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_ylabel("Monthly Charges ($)")
    fig.suptitle("Tenure & Spend Profiles", fontsize=18, fontweight="bold",
                 color=config.NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "03_tenure_and_charges.png")


def fig_correlation(df):
    """Correlation heatmap of numeric drivers vs. churn."""
    num = df.copy()
    num["ChurnFlag"] = (num[config.TARGET] == "Yes").astype(int)
    cols = ["ChurnFlag", "tenure", "MonthlyCharges", "TotalCharges",
            "NumAddOnServices", "AvgChargesPerMonth", "IsMonthToMonth",
            "IsElectronicCheck", "HasFiberOptic"]
    corr = num[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(cols, fontsize=10)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else config.NAVY, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation of Numeric Drivers with Churn",
                 fontsize=16, pad=14)
    _save(fig, "04_correlation_heatmap.png")


def fig_contract_payment_matrix(df):
    """Heatmap of churn rate across the contract x payment grid — a clear map
    of exactly which segment to target first."""
    pivot = df.assign(flag=(df[config.TARGET] == "Yes").astype(int)) \
              .pivot_table(index="Contract", columns="PaymentMethod",
                           values="flag", aggfunc="mean")
    pivot.columns = [c.replace(" (automatic)", "") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot, cmap="OrRd", vmin=0, vmax=pivot.values.max())
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="white" if v > 0.35 else config.NAVY,
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Churn rate")
    ax.set_title("Churn Rate: Contract Type x Payment Method", pad=14)
    _save(fig, "05_contract_payment_matrix.png")


def run_all_eda():
    config.set_theme()
    df = _load()
    print("[eda] generating figures...")
    fig_churn_overview(df)
    fig_churn_drivers(df)
    fig_tenure_charges(df)
    fig_correlation(df)
    fig_contract_payment_matrix(df)
    print(f"[eda] done -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    run_all_eda()
