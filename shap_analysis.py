"""
shap_analysis.py
-----------------
SHAP (SHapley Additive exPlanations) interpretability analysis for the
best-performing model (XGBoost), so project managers get not just a
risk score but WHY a project is flagged high-risk.

Outputs:
    - shap_summary_beeswarm.png : per-feature impact distribution
    - shap_feature_importance.png : mean |SHAP value| ranking
    - shap_dependence_cpi.png    : how CPI value drives risk score
    - shap_waterfall_example.png : explanation for one specific flagged project
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RESULTS, "figures")

with open(os.path.join(RESULTS, "models.pkl"), "rb") as f:
    bundle = pickle.load(f)

model = bundle["models"]["XGBoost"]
feature_cols = bundle["feature_cols"]
X_test = pd.read_csv(os.path.join(RESULTS, "X_test.csv"))
y_test = pd.read_csv(os.path.join(RESULTS, "y_test.csv")).iloc[:, 0]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Beeswarm summary plot
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title("SHAP Summary — Feature Impact on Overrun Risk Prediction", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_summary_beeswarm.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved shap_summary_beeswarm.png")

# 2. Mean |SHAP| feature importance bar chart
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
    "mean_abs_shap", ascending=True)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="#2E86AB")
ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
ax.set_title("SHAP Feature Importance — Overrun Risk Model", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_feature_importance.png"), dpi=160)
plt.close()
print("Saved shap_feature_importance.png")
importance_df.sort_values("mean_abs_shap", ascending=False).to_csv(
    os.path.join(RESULTS, "shap_feature_importance.csv"), index=False)

# 3. Dependence plot for CPI (the single most actionable EVM indicator)
plt.figure(figsize=(8, 6))
shap.dependence_plot("cpi", shap_values, X_test, feature_names=feature_cols, show=False,
                      interaction_index="spi")
plt.title("SHAP Dependence — CPI vs. Predicted Overrun Risk\n(colored by SPI)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_dependence_cpi.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved shap_dependence_cpi.png")

# 4. Waterfall explanation for one correctly-flagged high-risk project
proba = model.predict_proba(X_test)[:, 1]
candidates = np.where((y_test.values == 1) & (proba > 0.8))[0]
idx = candidates[0] if len(candidates) else 0
plt.figure(figsize=(9, 6))
explanation = shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value,
                                data=X_test.iloc[idx].values, feature_names=feature_cols)
shap.plots.waterfall(explanation, show=False, max_display=10)
plt.title(f"SHAP Waterfall — Example High-Risk Project\n(predicted overrun probability: {proba[idx]:.1%})",
          fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_waterfall_example.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved shap_waterfall_example.png")

print("\nTop 5 most important features by mean |SHAP|:")
print(importance_df.sort_values("mean_abs_shap", ascending=False).head(5).to_string(index=False))
