"""
business_impact.py
------------------
Turn the model's churn probabilities into a money story: revenue at risk, the
ROI of a targeted retention campaign, and a "who to call first" priority list.

Run directly:  python src/business_impact.py   (after train_model.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _load_scored() -> pd.DataFrame:
    path = config.MODELS_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("Run train_model.py first to create predictions.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Revenue at risk
# --------------------------------------------------------------------------- #
def revenue_at_risk(df: pd.DataFrame) -> dict:
    churners = df[df["ActualChurn"] == 1]
    monthly = churners["MonthlyCharges"].sum()
    annual = monthly * 12
    return {
        "n_churners": int(len(churners)),
        "monthly_revenue_at_risk": float(monthly),
        "annual_revenue_at_risk": float(annual),
        "avg_churner_value": float(churners["MonthlyCharges"].mean()),
    }


# --------------------------------------------------------------------------- #
# Campaign ROI from targeting the top-risk decile(s)
# --------------------------------------------------------------------------- #
def campaign_roi(df: pd.DataFrame, target_fraction: float = 0.20) -> dict:
    df = df.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)
    n_target = int(len(df) * target_fraction)
    targeted = df.head(n_target)

    true_churners = int(targeted["ActualChurn"].sum())
    saved = true_churners * config.RETENTION_CAMPAIGN_SUCCESS

    offer_cost = (n_target * config.AVG_RETENTION_OFFER_COST
                  * config.EXPECTED_LIFETIME_MONTHS)
    revenue_saved = (saved * targeted.loc[targeted["ActualChurn"] == 1,
                                          "MonthlyCharges"].mean()
                     * config.EXPECTED_LIFETIME_MONTHS)
    roi = (revenue_saved - offer_cost) / offer_cost if offer_cost else 0.0
    precision = true_churners / n_target if n_target else 0.0

    return {
        "target_fraction": target_fraction,
        "customers_targeted": n_target,
        "true_churners_caught": true_churners,
        "capture_precision": float(precision),
        "expected_customers_saved": float(saved),
        "campaign_cost": float(offer_cost),
        "revenue_protected": float(revenue_saved),
        "net_benefit": float(revenue_saved - offer_cost),
        "roi": float(roi),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_gain_chart(df: pd.DataFrame):
    """Cumulative-gains curve: by calling the riskiest X% of customers, what
    share of real churners do we reach? The lift vs. random is the value."""
    config.set_theme()
    d = df.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)
    cum = d["ActualChurn"].cumsum() / d["ActualChurn"].sum()
    pct = np.arange(1, len(d) + 1) / len(d)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(pct, cum, lw=3, color=config.TEAL, label="Model-guided targeting")
    ax.plot([0, 1], [0, 1], ls="--", color=config.SLATE, label="Random calling")
    for frac in (0.2, 0.4):
        reach = cum[int(len(d) * frac) - 1]
        ax.scatter([frac], [reach], color=config.CORAL, zorder=5, s=80)
        ax.annotate(f"Top {frac:.0%} -> {reach:.0%} of churners",
                    (frac, reach), textcoords="offset points", xytext=(10, -18),
                    fontsize=11, color=config.NAVY, fontweight="bold")
    ax.set_xlabel("Share of customer base contacted (riskiest first)")
    ax.set_ylabel("Share of actual churners reached")
    ax.set_title("Cumulative Gains — Targeting Efficiency")
    ax.legend(loc="lower right")
    fig.savefig(config.FIGURES_DIR / "09_cumulative_gains.png")
    plt.close(fig)
    print("[fig]  09_cumulative_gains.png")


def export_priority_list(df: pd.DataFrame, top_n: int = 250):
    """A ranked call-list the retention team could action tomorrow."""
    cols = [c for c in ["tenure", "Contract", "MonthlyCharges",
                        "InternetService", "PaymentMethod", "ChurnProbability",
                        "ActualChurn"] if c in df.columns]
    out = (df.sort_values("ChurnProbability", ascending=False)
             .head(top_n)[cols].reset_index(drop=True))
    out.index += 1
    out.insert(0, "Priority", out.index)
    path = config.MODELS_DIR / "retention_priority_list.csv"
    out.to_csv(path, index=False)
    print(f"[csv]  {path.name}  (top {top_n} at-risk customers)")


# --------------------------------------------------------------------------- #
def run():
    df = _load_scored()
    rar = revenue_at_risk(df)
    roi = campaign_roi(df)

    print("\n" + "=" * 60)
    print("  BUSINESS IMPACT (hold-out test set)")
    print("=" * 60)
    print(f"  Churners in test set      : {rar['n_churners']:,}")
    print(f"  Monthly revenue at risk   : ${rar['monthly_revenue_at_risk']:,.0f}")
    print(f"  Annualised revenue at risk: ${rar['annual_revenue_at_risk']:,.0f}")
    print(f"  Avg monthly value/churner : ${rar['avg_churner_value']:,.2f}")
    print("-" * 60)
    print(f"  Strategy: target riskiest {roi['target_fraction']:.0%} "
          f"({roi['customers_targeted']:,} customers)")
    print(f"  Real churners caught      : {roi['true_churners_caught']:,} "
          f"(precision {roi['capture_precision']:.0%})")
    print(f"  Expected customers saved  : {roi['expected_customers_saved']:.0f}")
    print(f"  Campaign cost             : ${roi['campaign_cost']:,.0f}")
    print(f"  Revenue protected         : ${roi['revenue_protected']:,.0f}")
    print(f"  Net benefit               : ${roi['net_benefit']:,.0f}")
    print(f"  ROI                       : {roi['roi']:.0%}")
    print("=" * 60 + "\n")

    fig_gain_chart(df)
    export_priority_list(df)

    pd.DataFrame([{**rar, **roi}]).to_csv(
        config.ROOT / "reports" / "business_impact.csv", index=False)
    return rar, roi


if __name__ == "__main__":
    run()
