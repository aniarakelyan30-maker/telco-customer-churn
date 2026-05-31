# CV / Portfolio Summary

Ready-to-paste blurbs describing this project at different lengths.

---

## 📌 One-line (CV bullet)

> **Customer Churn Prediction & Retention Strategy** — Built an end-to-end ML pipeline on 7K telecom customers (Python, scikit-learn) achieving **0.84 ROC-AUC / 78% recall**, and converted predictions into a targeted retention campaign with **~99% projected ROI**.

---

## 📌 Resume bullets (pick 2–3)

- Built a reproducible, end-to-end churn-prediction pipeline (cleaning → EDA → modeling → business impact) on the 7,043-customer IBM Telco dataset using **Python, pandas, and scikit-learn**.
- Engineered lifecycle and engagement features and compared Logistic Regression, Random Forest, and Gradient Boosting, selecting the model with the best **recall (0.78)** and **ROC-AUC (0.84)** under class imbalance.
- Identified the dominant churn drivers (month-to-month contracts churn **43%** vs **3%** for two-year), translating analysis into prioritized retention recommendations.
- Quantified business value with a cumulative-gains/ROI analysis: targeting the **riskiest 20%** of customers reaches **~50% of churners** and yields a **~99% campaign ROI**, delivered as an actionable call-list.

---

## 📌 Portfolio / LinkedIn paragraph

> I built an end-to-end customer-churn case study on the IBM Telco dataset (7,043 customers). After cleaning and feature engineering, I ran a full EDA that pinpointed the strongest churn drivers — contract type, payment method, internet service, and tenure. I then trained and compared three classifiers in a leakage-safe scikit-learn pipeline, selecting a Random Forest with **0.84 ROC-AUC and 78% recall** because catching churners matters more than raw accuracy. Finally, I converted the model into a business decision: a costed retention campaign that targets the riskiest 20% of customers to capture ~50% of all churners at roughly **99% ROI**, complete with a ranked call-list for the retention team. The project demonstrates both the technical pipeline and the business translation that turns a model into money.

---

## 📌 Talking points for interviews

- **Why recall over accuracy?** Missing a churner (losing a paying customer) costs far more than a wasted retention offer, so I optimized for recall and ROC-AUC rather than accuracy, and used `class_weight="balanced"`.
- **Why Random Forest over Gradient Boosting?** GB had higher accuracy but only ~50% recall; RF caught 78% of churners — the metric the business actually cares about.
- **How did you avoid data leakage?** Dropped the customer ID, fit all scaling/encoding inside a `Pipeline` on the training split only, and evaluated on a stratified hold-out.
- **How did you prove value?** A cumulative-gains curve plus an explicit, configurable ROI model (offer cost, save rate, value horizon) — not just abstract metrics.
