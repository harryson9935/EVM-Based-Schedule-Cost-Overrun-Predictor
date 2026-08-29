 EVM-Based Schedule/Cost Overrun Predictor

**Self Project | May 2025 – July 2025**

A predictive analytics framework that identifies schedule and cost
overrun risk in infrastructure projects by integrating **Earned Value
Management (EVM)** with machine learning — flagging high-risk projects
**2–3 milestones before** the overrun is obvious from the raw numbers,
with SHAP-based interpretability so the flag comes with a "why."

---

## 1. Objective

Develop a predictive analytics framework to identify schedule and cost
overrun risks in infrastructure projects by integrating Earned Value
Management (EVM) with machine learning.

## 2. Approach

- **Feature engineering:** the classic EVM indicators — **CPI, SPI,
  CV, SV, EAC, TCPI** — computed at every project milestone, plus
  3-period rolling mean/trend/volatility features on CPI and SPI (a
  single bad milestone matters less than a *worsening trend*).
- **Early-warning framing:** the training table only uses milestone
  data from the first ~79% of each project's timeline (through
  milestone 11 of 14) to predict the project's **final** outcome —
  so a model trained on it is, by construction, forecasting overrun
  risk **2–3 milestones ahead** of it happening.
- **Models compared:** Logistic Regression (interpretable baseline),
  Random Forest, and XGBoost — trained with `scikit-learn` /
  `xgboost`, evaluated with a **project-grouped** train/test split
  (no milestone from the same project appears in both train and test).
- **Interpretability:** SHAP (`TreeExplainer`) on the best model
  (XGBoost) to rank feature importance and explain individual
  predictions for project managers.

## 3. Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **79.6%** | **0.908** | 0.899 | 0.722 | 0.801 |
| Random Forest | 79.3% | 0.903 | 0.901 | 0.715 | 0.797 |
| Logistic Regression | 79.2% | 0.871 | 0.886 | 0.727 | 0.799 |

*(Held-out test set, 65 projects / 715 milestone-rows never seen during training. XGBoost 5-fold grouped cross-validation: ROC-AUC = 0.895 ± 0.020, confirming the result isn't a lucky split.)*

This lands close to the target performance band (~84% accuracy /
0.86 AUC) referenced in the project brief — XGBoost modestly
outperforms both baselines, and all three models comfortably beat
random guessing (AUC 0.5) with 2-3 milestones of lead time.

**Top risk indicators (by mean |SHAP value|):** `cpi_roll_mean3`
(3-period rolling CPI) dominates, followed by raw `cpi`,
`cv_pct_bac` (cost variance as % of budget), `actual_pct_complete`,
and `spi`. In other words: **a worsening CPI trend is a stronger
early-warning signal than any single-milestone snapshot** — exactly
the insight a PM can act on.

## 4. Visualizations

**CPI/SPI trend divergence — overrun vs. on-track projects:**
![Trend Divergence](results/figures/evm_trend_divergence.png)

**Model comparison:**
![Model Comparison](results/figures/model_comparison.png)

**ROC curves:**
![ROC Curves](results/figures/roc_curves.png)

**Confusion matrices:**
![Confusion Matrices](results/figures/confusion_matrices.png)

**SHAP feature importance:**
![SHAP Importance](results/figures/shap_feature_importance.png)

**SHAP summary (beeswarm):**
![SHAP Summary](results/figures/shap_summary_beeswarm.png)

**SHAP dependence — CPI (colored by SPI):**
![SHAP Dependence](results/figures/shap_dependence_cpi.png)

**SHAP waterfall — example flagged high-risk project:**
![SHAP Waterfall](results/figures/shap_waterfall_example.png)

## 5. Repository Structure

```
evm-overrun-predictor/
├── README.md
├── requirements.txt
├── data/
│   ├── evm_full_timeseries.csv       # All 260 projects × 14 milestones (full lifecycle)
│   └── evm_milestone_features.csv    # Training view: milestones ≤11 only (early-warning cutoff)
├── src/
│   ├── generate_data.py               # Synthetic EVM data generator (risk-driven drift simulation)
│   ├── train_models.py                # Trains/compares LogReg, RF, XGBoost; grouped CV; figures
│   ├── shap_analysis.py               # SHAP interpretability on the best model
│   └── visualize_trends.py            # CPI/SPI trajectory divergence plot
├── notebooks/
│   └── exploration.ipynb              # Interactive walkthrough
└── results/
    ├── model_comparison.csv
    ├── cv_summary.json
    ├── classification_reports.txt
    ├── shap_feature_importance.csv
    ├── models.pkl                     # Pickled fitted models + scaler
    ├── X_test.csv / y_test.csv
    └── figures/*.png
```

## 6. How to Run

```bash
pip install -r requirements.txt

python src/generate_data.py       # 1. Build synthetic EVM dataset
python src/train_models.py        # 2. Train & compare all 3 models, produce metrics + figures
python src/shap_analysis.py       # 3. SHAP interpretability on the best model
python src/visualize_trends.py    # 4. CPI/SPI trend divergence chart
```

## 7. Methodology Detail

**EVM indicators used as features (per milestone):**

| Indicator | Formula | Interpretation |
|---|---|---|
| CPI | EV / AC | Cost efficiency (>1 = under budget) |
| SPI | EV / PV | Schedule efficiency (>1 = ahead of schedule) |
| CV | EV − AC | Cost variance ($) |
| SV | EV − PV | Schedule variance ($) |
| EAC | BAC / CPI | Forecast total cost at completion |
| TCPI | (BAC − EV) / (BAC − AC) | Efficiency required on remaining work to hit budget |

**Engineered trend features:** 3-period rolling mean and standard
deviation of CPI/SPI, and a 2-lag trend (slope) on each — computed
**causally** (only using data up to and including the current
milestone) to avoid any look-ahead leakage.

**Label construction:** a project is labeled `overrun = 1` if its
**final** milestone CPI < 0.92 or SPI < 0.86; this label is then
attached to every earlier milestone row for that project up to the
early-warning cutoff (milestone 11 of 14) — so every training example
is "predict the eventual outcome using only what was knowable 2–3
milestones earlier."

**Validation:** `GroupShuffleSplit` and `StratifiedGroupKFold`
grouped by `project_id` ensure milestones from the same project never
appear in both train and test, which would otherwise leak information
and inflate performance.

## 8. Data Disclaimer

All data is **synthetically generated** (`src/generate_data.py`,
seeded for reproducibility) — 260 simulated infrastructure projects
(highways, bridges, water treatment plants, transmission lines,
airports, rail, dams, hospitals) across 5 U.S. regions, with hidden
risk factors (contractor quality, site risk, regulatory risk, supply
chain exposure, design complexity) driving realistic cost/schedule
drift plus reporting noise. No proprietary or confidential project
data is used.

## 9. Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `pandas` / `NumPy` ·
`Matplotlib`

## 10. License

MIT
