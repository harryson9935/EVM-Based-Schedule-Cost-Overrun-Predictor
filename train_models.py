"""
train_models.py
----------------
Trains and compares three classifiers to predict eventual project cost/
schedule overrun from milestone-level EVM features, observed 2-3
milestones before project completion (see generate_data.py for the
lookahead framing):

    - Logistic Regression (baseline, interpretable linear model)
    - Random Forest
    - XGBoost

Evaluation uses a GROUP-AWARE split (by project_id) so no project's
milestones leak across train/test -- this is essential since milestones
from the same project are highly correlated.

Outputs: metrics table, confusion matrices, ROC curves, and the fitted
models (pickled) for use by shap_analysis.py.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
import pickle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

FEATURE_COLS = [
    "milestone", "planned_pct_complete", "actual_pct_complete",
    "cpi", "spi", "cv_pct_bac", "sv_pct_bac", "tcpi",
    "cpi_roll_mean3", "spi_roll_mean3", "cpi_roll_std3", "spi_roll_std3",
    "cpi_trend3", "spi_trend3", "had_shock",
]
TARGET_COL = "final_overrun"
GROUP_COL = "project_id"


def load_training_data():
    df = pd.read_csv(os.path.join(DATA, "evm_milestone_features.csv"))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def split_train_test(df, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }, proba, pred


def main():
    df = load_training_data()
    train_df, test_df = split_train_test(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    print(f"Train rows: {len(X_train)} ({train_df[GROUP_COL].nunique()} projects) | "
          f"Test rows: {len(X_test)} ({test_df[GROUP_COL].nunique()} projects)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = []
    probas = {}
    preds = {}

    # 1. Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    lr.fit(X_train_scaled, y_train)
    m, p, pr = evaluate(lr, X_test_scaled, y_test, "Logistic Regression")
    results.append(m); probas["Logistic Regression"] = p; preds["Logistic Regression"] = pr
    models["Logistic Regression"] = lr

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m, p, pr = evaluate(rf, X_test, y_test, "Random Forest")
    results.append(m); probas["Random Forest"] = p; preds["Random Forest"] = pr
    models["Random Forest"] = rf

    # 3. XGBoost
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    m, p, pr = evaluate(xgb, X_test, y_test, "XGBoost")
    results.append(m); probas["XGBoost"] = p; preds["XGBoost"] = pr
    models["XGBoost"] = xgb

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)
    print("\n=== Model Comparison (held-out test set) ===")
    print(results_df.to_string(index=False))

    # 5-fold grouped CV for the best model (robustness check)
    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model by ROC-AUC: {best_name}")

    sgkf = StratifiedGroupKFold(n_splits=5)
    cv_aucs = []
    Xg = df[FEATURE_COLS].reset_index(drop=True)
    yg = df[TARGET_COL].reset_index(drop=True)
    groups = df[GROUP_COL].reset_index(drop=True)
    for tr_idx, te_idx in sgkf.split(Xg, yg, groups=groups):
        xgb_cv = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
                                random_state=42, n_jobs=-1)
        xgb_cv.fit(Xg.iloc[tr_idx], yg.iloc[tr_idx])
        p = xgb_cv.predict_proba(Xg.iloc[te_idx])[:, 1]
        cv_aucs.append(roc_auc_score(yg.iloc[te_idx], p))
    print(f"XGBoost 5-fold grouped CV ROC-AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")
    with open(os.path.join(RESULTS, "cv_summary.json"), "w") as f:
        json.dump({"xgboost_5fold_cv_auc_mean": float(np.mean(cv_aucs)),
                    "xgboost_5fold_cv_auc_std": float(np.std(cv_aucs)),
                    "fold_aucs": [float(x) for x in cv_aucs]}, f, indent=2)

    # Save classification reports
    with open(os.path.join(RESULTS, "classification_reports.txt"), "w") as f:
        for name in models:
            f.write(f"=== {name} ===\n")
            f.write(classification_report(y_test, preds[name]))
            f.write("\n\n")

    # --- Figures ---
    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = {"Logistic Regression": "#888888", "Random Forest": "#F6A800", "XGBoost": "#2E86AB"}
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        auc = results_df[results_df.model == name]["roc_auc"].values[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Overrun Prediction\n(2-3 milestones ahead of project completion)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=160)
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, models):
        cm = confusion_matrix(y_test, preds[name])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["On Track", "Overrun"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["On Track", "Overrun"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        acc = results_df[results_df.model == name]["accuracy"].values[0]
        ax.set_title(f"{name}\n(Accuracy={acc:.1%})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion_matrices.png"), dpi=160)
    plt.close()

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w/2, results_df["accuracy"], width=w, label="Accuracy", color="#2E86AB")
    ax.bar(x + w/2, results_df["roc_auc"], width=w, label="ROC-AUC", color="#F6A800")
    ax.set_xticks(x); ax.set_xticklabels(results_df["model"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Accuracy & ROC-AUC (held-out test set)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    for i, (a, auc) in enumerate(zip(results_df["accuracy"], results_df["roc_auc"])):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, auc + 0.01, f"{auc:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=160)
    plt.close()

    # Save models + scaler + test set for downstream SHAP analysis
    with open(os.path.join(RESULTS, "models.pkl"), "wb") as f:
        pickle.dump({"models": models, "scaler": scaler, "feature_cols": FEATURE_COLS}, f)
    X_test.to_csv(os.path.join(RESULTS, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(RESULTS, "y_test.csv"), index=False)

    print("\nSaved: model_comparison.csv, cv_summary.json, classification_reports.txt,")
    print("       figures/roc_curves.png, figures/confusion_matrices.png, figures/model_comparison.png")
    print("       models.pkl, X_test.csv, y_test.csv")


if __name__ == "__main__":
    main()
