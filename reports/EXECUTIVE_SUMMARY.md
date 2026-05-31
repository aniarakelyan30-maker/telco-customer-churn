# Executive Summary — Customer Churn

**Audience:** business stakeholders (no code).
**Bottom line:** churn is a ~$1.6M/year revenue problem, it is highly predictable, and a focused retention play pays for itself roughly 2×.

---

## The situation
- **26% of customers churn.** They are disproportionately **new** (47% churn in year one) and **higher-paying**, so the revenue impact exceeds the headcount impact.
- Churn is **concentrated, not random** — it clusters in identifiable segments, which means it is addressable.

## What's driving it
| Driver | Finding | Implication |
|---|---|---|
| Contract | Month-to-month **43%** vs two-year **3%** | Commitment is the single biggest lever |
| Internet | Fiber optic **42%** vs DSL **19%** | Price/quality perception issue in fiber |
| Payment | Electronic check **45%** vs auto-pay ~15% | Payment friction signals disengagement |
| Tenure | First year **47%**, drops sharply after | Onboarding is make-or-break |

## What we can do about it
We can **predict churn before it happens** with a model that catches **78% of churners** (ROC-AUC 0.84). Rather than discount everyone, we rank customers by risk and act on the top slice:

- Contact the **riskiest 20%** → reach **~50% of all churners** (2.5× better than random).
- Expected outcome on a 1,405-customer test slice: **~56 customers saved, ~$54k revenue protected, ~$27k net benefit, ~99% ROI.**
- Scaled across the full base, this is a six-figure annual opportunity.

## Recommended actions (priority order)
1. Migrate month-to-month customers to annual contracts via targeted loyalty offers.
2. Audit fiber-optic pricing and service quality.
3. Shift electronic-check payers to auto-pay.
4. Strengthen first-year onboarding and proactive support.
5. Run the churn model monthly and feed the retention team a ranked call-list.

*Underlying assumptions (offer cost, save rate, value horizon) are explicit and adjustable — happy to re-run under your numbers.*
