"""
train_model.py
--------------
Train and compare three classifiers for churn prediction, pick the best by
ROC-AUC, and produce evaluation artefacts: a metrics table, ROC curves, a
confusion matrix, and a feature-importance chart.

Run directly:  python src/train_model.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from . import config
except ImportError:
    import config


# --------------------------------------------------------------------------- #
# Data prep for modelling
# --------------------------------------------------------------------------- #
def load_xy():
    df = pd.read_csv(config.PROCESSED_DATA)
    y = (df[config.TARGET] == "Yes").astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y


def build_preprocessor(X):
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
    ])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1,
            random_state=config.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=config.RANDOM_STATE),
    }


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate():
    config.set_theme()
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_STATE)
    print(f"[split] train={len(X_train):,}  test={len(X_test):,}  "
          f"churn(train)={y_train.mean():.1%}")

    pre = build_preprocessor(X)
    results, fitted, roc_data = [], {}, {}

    for name, clf in candidate_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })
        fitted[name] = pipe
        roc_data[name] = roc_curve(y_test, proba)
        print(f"[fit]   {name:<20} AUC={results[-1]['ROC_AUC']:.3f}  "
              f"Recall={results[-1]['Recall']:.3f}")

    metrics = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    metrics.to_csv(config.METRICS_FILE, index=False)
    print(f"\n[metrics]\n{metrics.round(3).to_string(index=False)}")

    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\n[best]  {best_name}")
    print(classification_report(y_test, best_model.predict(X_test),
                                target_names=["Retained", "Churned"]))

    # Persist best model + the test split for downstream business analysis.
    joblib.dump(best_model, config.MODELS_DIR / "best_model.joblib")
    best_metrics = {k: round(float(v), 4)
                    for k, v in metrics.iloc[0].drop("Model").items()}
    (config.MODELS_DIR / "best_model.json").write_text(
        json.dumps({"model": best_name, "metrics": best_metrics}, indent=2))

    _plot_roc(roc_data, metrics)
    _plot_confusion(y_test, best_model.predict(X_test), best_name)
    _plot_feature_importance(best_model, best_name)
    _export_test_predictions(best_model, X_test, y_test)
    return metrics, best_name


# --------------------------------------------------------------------------- #
# Evaluation figures
# --------------------------------------------------------------------------- #
def _plot_roc(roc_data, metrics):
    fig, ax = plt.subplots(figsize=(9, 8))
    palette = [config.CORAL, config.TEAL, config.AMBER]
    auc_map = dict(zip(metrics["Model"], metrics["ROC_AUC"]))
    for (name, (fpr, tpr, _)), color in zip(roc_data.items(), palette):
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{name} (AUC = {auc_map[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=config.SLATE, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=11)
    fig.savefig(config.FIGURES_DIR / "06_roc_curves.png")
    plt.close(fig)
    print("[fig]  06_roc_curves.png")


def _plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j]/total:.1%})",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else config.NAVY)
    ax.set_title(f"Confusion Matrix — {name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(config.FIGURES_DIR / "07_confusion_matrix.png")
    plt.close(fig)
    print("[fig]  07_confusion_matrix.png")


def _feature_names(model):
    return model.named_steps["pre"].get_feature_names_out()


def _plot_feature_importance(model, name, top_n=15):
    names = _feature_names(model)
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        xlabel = "Importance"
    else:
        imp = np.abs(clf.coef_[0])
        xlabel = "|Coefficient|"
    order = np.argsort(imp)[::-1][:top_n]
    clean = [n.split("__", 1)[-1] for n in names[order]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(order)), imp[order][::-1], color=config.NAVY)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(clean[::-1], fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} Churn Predictors — {name}")
    fig.savefig(config.FIGURES_DIR / "08_feature_importance.png")
    plt.close(fig)
    print("[fig]  08_feature_importance.png")


def _export_test_predictions(model, X_test, y_test):
    """Score the hold-out set so the business module can value the model."""
    out = X_test.copy()
    out["ActualChurn"] = y_test.values
    out["ChurnProbability"] = model.predict_proba(X_test)[:, 1]
    out.to_csv(config.MODELS_DIR / "test_predictions.csv", index=False)


if __name__ == "__main__":
    train_and_evaluate()
