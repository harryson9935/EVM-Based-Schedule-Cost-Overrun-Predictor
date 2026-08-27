# EVM-Based Schedule/Cost Overrun Predictor

**Self Project | May 2025 – July 2025**

## Overview

The **EVM-Based Schedule/Cost Overrun Predictor** is a predictive analytics framework designed to identify **schedule and cost overrun risks in infrastructure projects at an early stage**.

Traditional Earned Value Management (EVM) provides powerful indicators of project cost and schedule performance, but conventional EVM-based forecasting can be limited when project conditions change over time or when multiple performance indicators interact in nonlinear ways.

This project combines **Earned Value Management metrics with machine learning classification models** to improve early risk identification.

Key EVM indicators including:

* Cost Performance Index (CPI)
* Schedule Performance Index (SPI)
* Cost Variance (CV)
* Schedule Variance (SV)
* Estimate at Completion (EAC)
* To-Complete Performance Index (TCPI)

were engineered across project milestones and used as predictive features.

Multiple classification algorithms were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

**SHAP (SHapley Additive Explanations)** was used to interpret model predictions and identify the EVM indicators contributing most strongly to project risk.

The final framework achieved approximately **84% classification accuracy** and **0.86 ROC-AUC**, enabling identification of high-risk project phases approximately **2–3 milestones ahead**.

---

# Objective

The primary objective was to develop a predictive analytics framework capable of identifying potential **schedule and cost overruns before they become critical**.

The project aimed to:

1. Integrate EVM metrics into a machine learning pipeline.
2. Engineer project performance indicators across milestones.
3. Identify early warning signals for cost and schedule risk.
4. Train multiple classification algorithms.
5. Compare model performance using appropriate classification metrics.
6. Identify the best-performing predictive model.
7. Use SHAP for model interpretability.
8. Predict high-risk project phases 2–3 milestones ahead.
9. Provide interpretable risk indicators for project managers.
10. Support proactive project monitoring and corrective action.

---

# Problem Statement

Infrastructure projects frequently experience:

* Cost overruns
* Schedule delays
* Resource constraints
* Procurement delays
* Productivity losses
* Scope changes
* Unexpected risks

Traditional project monitoring often identifies these issues after they have already affected project performance.

The goal of this project was to shift the monitoring approach from:

```text id="uk8k1q"
Performance Declines
       ↓
Problem Becomes Visible
       ↓
Management Response
```

toward:

```text id="e4z0tb"
Current Project Data
       ↓
EVM Metrics
       ↓
Machine Learning Model
       ↓
Early Risk Prediction
       ↓
Management Intervention
       ↓
Reduced Overrun Risk
```

This transforms EVM from a primarily descriptive monitoring framework into a **predictive early-warning system**.

---

# Earned Value Management

Earned Value Management provides a structured method for measuring project performance by comparing:

* Planned work
* Completed work
* Actual expenditure

Three fundamental quantities are:

### Planned Value (PV)

The budgeted value of work that was scheduled to be completed by a particular point in time.

### Earned Value (EV)

The budgeted value of work that has actually been completed.

### Actual Cost (AC)

The actual cost incurred for the completed work.

These values form the basis for the EVM indicators used in the machine learning model.

---

# EVM Metrics Used

## 1. Cost Variance

Cost Variance measures the difference between earned value and actual cost.

$$
CV = EV - AC
$$

Interpretation:

```text id="z5h7op"
CV > 0 → Under Budget
CV = 0 → On Budget
CV < 0 → Over Budget
```

A negative CV indicates that the project has spent more than the budgeted value of completed work.

---

# 2. Schedule Variance

Schedule Variance measures the difference between earned value and planned value.

$$
SV = EV - PV
$$

Interpretation:

```text id="5ey7yl"
SV > 0 → Ahead of Schedule
SV = 0 → On Schedule
SV < 0 → Behind Schedule
```

A negative SV indicates that less work has been completed than originally planned.

---

# 3. Cost Performance Index

Cost Performance Index measures cost efficiency.

$$
CPI = \frac{EV}{AC}
$$

Interpretation:

```text id="a9fvqw"
CPI > 1 → Better Cost Performance
CPI = 1 → On Budget
CPI < 1 → Cost Inefficiency
```

For example:

$$
CPI=0.85
$$

means that the project is generating approximately ₹0.85 worth of earned value for every ₹1 spent, under the chosen units of measurement.

---

# 4. Schedule Performance Index

Schedule Performance Index measures schedule efficiency.

$$
SPI = \frac{EV}{PV}
$$

Interpretation:

```text id="0txb7t"
SPI > 1 → Ahead of Schedule
SPI = 1 → On Schedule
SPI < 1 → Behind Schedule
```

For example:

$$
SPI=0.80
$$

indicates that the project is progressing at approximately 80% of the planned rate according to the EVM measurement.

---

# 5. Estimate at Completion

Estimate at Completion (EAC) estimates the expected total project cost based on current performance.

One common formulation is:

$$
EAC = \frac{BAC}{CPI}
$$

where:

* BAC = Budget at Completion
* CPI = Cost Performance Index

The formulation can be adapted depending on assumptions about future project performance.

EAC provides a forward-looking indicator of potential final cost.

---

# 6. To-Complete Performance Index

TCPI measures the cost efficiency required to achieve a specified project financial target.

For the original budget:

$$
TCPI = \frac{BAC-EV}{BAC-AC}
$$

For an updated estimate at completion:

$$
TCPI = \frac{BAC-EV}{EAC-AC}
$$

A high TCPI can indicate that substantial improvement in future cost efficiency is required to meet the desired financial target.

---

# Feature Engineering

The EVM indicators were calculated across project milestones to convert project performance information into structured machine learning features.

A simplified dataset structure is:

| Project | Milestone |  CPI |  SPI | CV | SV | EAC | TCPI | Risk |
| ------- | --------: | ---: | ---: | -: | -: | --: | ---: | ---- |
| P1      |         1 | 0.96 | 0.98 |  - |  - |   - |    - | Low  |
| P1      |         2 | 0.91 | 0.93 |  - |  - |   - |    - | Low  |
| P1      |         3 | 0.84 | 0.87 |  - |  - |   - |    - | High |
| P2      |         1 | 1.02 | 1.01 |  - |  - |   - |    - | Low  |
| P2      |         2 | 0.89 | 0.82 |  - |  - |   - |    - | High |

The exact target definition depends on the project dataset and forecasting horizon.

---

# Prediction Target

The machine learning problem was formulated as a **binary classification task**.

The model predicts whether a future project phase/milestone is likely to be classified as:

```text id="q2v2aa"
0 → Low / Normal Risk

1 → High Risk
```

The target was designed around future schedule/cost performance rather than simply reproducing the current EVM status.

This distinction is important because the objective is **early prediction**, not merely detection of an already-known overrun.

---

# Early Prediction Framework

A key component of the project was forecasting risk several milestones ahead.

Instead of:

```text id="ihq5n6"
Milestone t
    ↓
Predict Risk at t
```

the framework follows:

```text id="y8eg7a"
Milestone t
    ↓
EVM Features at t
    ↓
Machine Learning Model
    ↓
Risk at t + 2 / t + 3
```

This provides project managers with an opportunity to intervene before the predicted risk materializes.

---

# Machine Learning Models

Three classification approaches were implemented and compared.

---

# 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

The probability of the positive class is modeled as:

$$
P(Y=1)=\frac{1}{1+e^{-z}}
$$

where:

$$
z=\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n
$$

Logistic Regression provides a simple and interpretable benchmark and helps establish whether nonlinear models provide meaningful performance improvements.

---

# 2. Random Forest

Random Forest was implemented to capture nonlinear relationships between EVM indicators and future project risk.

Random Forest combines predictions from multiple decision trees.

Conceptually:

```text id="3m4t9n"
                Dataset
                   |
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      Tree 1     Tree 2     Tree N
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
             Final Prediction
```

Advantages include:

* Nonlinear relationship modeling
* Robustness to complex feature interactions
* Reduced overfitting compared with a single decision tree
* Feature importance estimation
* Good performance on structured/tabular data

---

# 3. XGBoost

XGBoost was used as a high-performance gradient-boosting classifier.

The algorithm sequentially builds trees where later trees focus on correcting errors made by earlier trees.

A simplified boosting process is:

```text id="7tq9wp"
Initial Prediction
       ↓
Calculate Errors
       ↓
Train Next Tree
       ↓
Update Prediction
       ↓
Calculate Remaining Errors
       ↓
Train Next Tree
       ↓
Final Prediction
```

XGBoost is particularly effective for structured/tabular datasets and can capture complex nonlinear interactions among EVM indicators.

---

# Model Development Pipeline

The machine learning pipeline followed:

```text id="2r0i7k"
Project Data
     ↓
EVM Calculation
     ↓
Feature Engineering
     ↓
Target Construction
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Model Training
     ↓
Cross-Validation
     ↓
Hyperparameter Optimization
     ↓
Model Comparison
     ↓
Final Model
     ↓
SHAP Interpretation
     ↓
Early Risk Prediction
```

---

# Train-Test Strategy

The dataset was separated into training and testing subsets.

The training set was used for:

* Model fitting
* Cross-validation
* Hyperparameter tuning

The test set was reserved for evaluating final generalization performance.

For project milestone data, care must be taken to avoid **data leakage**, particularly when multiple observations belong to the same project.

A robust implementation should ensure that information from future milestones does not enter the training features used to predict earlier milestones.

---

# Model Evaluation

The models were evaluated using several classification metrics.

## Accuracy

Accuracy measures the proportion of correctly classified observations.

$$
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
$$

The final model achieved approximately:

### **84% Classification Accuracy**

---

# ROC-AUC

The Receiver Operating Characteristic Area Under the Curve measures the model's ability to distinguish between high-risk and low-risk observations across classification thresholds.

The ROC curve plots:

$$
TPR = \frac{TP}{TP+FN}
$$

against:

$$
FPR = \frac{FP}{FP+TN}
$$

The model achieved:

### **ROC-AUC = 0.86**

A higher ROC-AUC indicates stronger discrimination between the two classes.

An AUC of 0.86 indicates that the model has strong ability to rank high-risk observations above low-risk observations across different classification thresholds.

---

# Confusion Matrix

The confusion matrix provides a detailed view of classification outcomes.

```text id="c9ycrx"
                         Predicted
                    Low Risk   High Risk
Actual Low Risk        TN         FP
Actual High Risk       FN         TP
```

Where:

* **TP** = Correctly identified high-risk projects
* **TN** = Correctly identified low-risk projects
* **FP** = Low-risk projects incorrectly flagged as high-risk
* **FN** = High-risk projects incorrectly classified as low-risk

For project management applications, false negatives can be particularly important because they represent risky project phases that the model failed to flag.

---

# Precision, Recall and F1-Score

Additional evaluation metrics can be used to assess the model.

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

Measures how many predicted high-risk cases were actually high risk.

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

Measures how many actual high-risk cases were successfully identified.

### F1-Score

$$
F1=2\times
\frac{Precision\times Recall}
{Precision+Recall}
$$

F1-score provides a balance between precision and recall.

---

# Explainable AI with SHAP

A major component of the project was **model interpretability**.

Complex machine learning models such as Random Forest and XGBoost can produce accurate predictions but may be difficult to interpret directly.

**SHAP (SHapley Additive exPlanations)** was used to explain model predictions.

SHAP assigns an importance value to each feature for a particular prediction.

Conceptually:

```text id="prz7t2"
Project Risk Prediction
          |
          ↓
     Model Output
          |
     ┌────┼────┐
     ↓    ↓    ↓
    CPI  SPI  EAC
     ↓    ↓    ↓
   Risk Contribution
```

---

# SHAP Feature Interpretation

SHAP values can indicate whether a feature pushes a prediction toward:

```text
Low Risk ←────────────→ High Risk
```

For example, a low CPI or SPI may contribute positively toward a high-risk prediction.

The SHAP framework therefore provides two levels of interpretation:

### Global Interpretation

Which EVM variables are most influential across the entire dataset?

### Local Interpretation

Why did the model classify a specific project milestone as high risk?

---

# Example Risk Interpretation

A project manager could receive an output such as:

```text id="jv1jcx"
Predicted Risk: HIGH

Key Risk Drivers:

1. SPI
   Strong contribution toward high risk

2. CPI
   Moderate contribution toward high risk

3. EAC
   Increased projected final cost

4. TCPI
   Indicates higher required future efficiency
```

This is more actionable than simply receiving:

```text
Risk = High
```

because the manager can understand which project performance indicators are driving the prediction.

---

# Early Warning System

The proposed framework can be integrated into a project monitoring process.

```text id="ytf99n"
Milestone Data
      ↓
Calculate EVM Metrics
      ↓
CPI / SPI / CV / SV / EAC / TCPI
      ↓
Machine Learning Model
      ↓
Risk Probability
      ↓
┌───────────────────────┐
│ High-Risk Prediction? │
└───────────┬───────────┘
            │
       ┌────┴────┐
       ↓         ↓
      No        Yes
       ↓         ↓
 Continue     Investigate
 Monitoring      ↓
             Corrective
               Action
```

The framework enables project teams to identify potential problems **2–3 milestones before the predicted high-risk phase**.

---

# Results

The developed predictive framework achieved approximately:

| Metric                  |                                      Result |
| ----------------------- | ------------------------------------------: |
| Classification Accuracy |                                     **84%** |
| ROC-AUC                 |                                    **0.86** |
| Prediction Horizon      |                    **2–3 milestones ahead** |
| Models Evaluated        | Logistic Regression, Random Forest, XGBoost |
| Explainability          |                                        SHAP |
| Feature Domain          |             EVM Project Performance Metrics |

The results demonstrate that combining traditional EVM indicators with machine learning can provide an effective early-warning mechanism for infrastructure project risk.

---

# Key Findings

## 1. EVM metrics contain predictive information

Indicators such as CPI, SPI, EAC, and TCPI provide useful signals regarding future project performance.

## 2. Nonlinear models can capture complex relationships

Random Forest and XGBoost can model interactions among EVM indicators that may not be captured by Logistic Regression.

## 3. Early prediction is more useful than retrospective detection

The primary value of the framework comes from predicting risk several milestones before the affected phase rather than simply identifying an overrun after it occurs.

## 4. ROC-AUC provides useful discrimination insight

The ROC-AUC of **0.86** indicates strong separation between high-risk and low-risk project phases across classification thresholds.

## 5. Explainability improves managerial usefulness

SHAP allows project managers to understand which EVM indicators are contributing to a high-risk prediction.

## 6. The framework supports proactive project management

Instead of waiting for cost or schedule overruns to become severe, project managers can use predicted risk signals to investigate and take corrective action earlier.

---

# Example Decision Framework

The model can be integrated into project governance as follows:

| Predicted Risk | Suggested Management Response           |
| -------------- | --------------------------------------- |
| Low            | Continue standard monitoring            |
| Moderate       | Increase monitoring frequency           |
| High           | Investigate root causes                 |
| Very High      | Initiate corrective action / escalation |

The exact thresholds should be determined based on project-specific risk tolerance and validation results.

---

# Business / Operational Value

The project demonstrates how predictive analytics can support infrastructure project management by:

* Providing early warning of potential overruns.
* Prioritizing high-risk project phases.
* Supporting proactive resource allocation.
* Improving project monitoring.
* Helping identify deteriorating cost performance.
* Helping identify deteriorating schedule performance.
* Providing interpretable risk drivers.
* Supporting data-driven management decisions.

---

# Project Structure

A recommended GitHub repository structure is:

```text id="m8hz0k"
EVM-Based-Schedule-Cost-Overrun-Predictor/
│
├── data/
│   ├── raw/
│   │   └── project_data.csv
│   │
│   └── processed/
│       └── evm_features.csv
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EVM_Feature_Engineering.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Comparison.ipynb
│   └── 06_SHAP_Analysis.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── evm_metrics.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── explainability.py
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── predictions.csv
│   └── shap_plots/
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Model development                |
| Pandas           | Data manipulation                |
| NumPy            | Numerical computation            |
| Scikit-learn     | Machine learning                 |
| XGBoost          | Gradient-boosting classification |
| SHAP             | Model interpretability           |
| Matplotlib       | Visualization                    |
| Seaborn          | Statistical visualization        |
| Jupyter Notebook | Interactive analysis             |

---

# Installation

Clone the repository:

```bash id="j6vxxs"
git clone https://github.com/<your-username>/EVM-Based-Schedule-Cost-Overrun-Predictor.git
```

Navigate to the project directory:

```bash id="r9t4pn"
cd EVM-Based-Schedule-Cost-Overrun-Predictor
```

Install the required dependencies:

```bash id="l7qtxc"
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash id="2qf8h9"
jupyter notebook
```

---

# Requirements

Example `requirements.txt`:

```text id="7ecx7x"
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
```

---

# Reproducibility

The analysis can be reproduced using the following workflow:

1. Load the project milestone dataset.
2. Clean and preprocess the project data.
3. Calculate PV, EV, and AC where required.
4. Engineer CPI, SPI, CV, SV, EAC, and TCPI.
5. Construct the future-risk target.
6. Split the data into training and testing sets.
7. Train Logistic Regression.
8. Train Random Forest.
9. Train XGBoost.
10. Compare model performance.
11. Tune model hyperparameters where required.
12. Evaluate the final model using accuracy, ROC-AUC, confusion matrix, precision, recall, and F1-score.
13. Apply SHAP to interpret model predictions.
14. Generate early-warning predictions for future project milestones.

---

# Recommended Visualizations

The repository can include the following visualizations.

## 1. CPI Trend

Plot CPI across project milestones.

```text id="l4m9u7"
CPI
 ↑
1.2 |───────
1.0 |───────╲
0.8 |        ╲────
0.6 |
    └────────────────→
       Milestones
```

A declining CPI can indicate deteriorating cost efficiency.

---

## 2. SPI Trend

Visualize schedule performance across milestones.

A persistent SPI below 1 can indicate that project progress is behind plan.

---

## 3. Cost and Schedule Variance

Plot CV and SV over time to identify periods of negative performance.

---

## 4. EAC Forecast

Compare the estimated final project cost against the original Budget at Completion.

```text id="9yl5c4"
Cost
 ↑
 │             EAC
 │            /
 │           /
 │----------/---- BAC
 │         /
 │        /
 └────────────────→
        Time
```

---

## 5. Confusion Matrix

Display the final model's classification performance.

---

## 6. ROC Curve

Plot the True Positive Rate against the False Positive Rate across different classification thresholds.

---

## 7. SHAP Summary Plot

Visualize the relative contribution of EVM features to model predictions.

This can reveal whether variables such as CPI, SPI, EAC, or TCPI are the strongest predictors of project risk.

---

# Testing Strategy

The model should be tested under multiple conditions.

### Normal Project Performance

```text id="f8q3jz"
CPI ≈ 1
SPI ≈ 1
     ↓
Expected Low Risk
```

### Cost Deterioration

```text id="0i8c1f"
CPI ↓
CV ↓
EAC ↑
     ↓
Increasing Cost Risk
```

### Schedule Deterioration

```text id="2l6i4e"
SPI ↓
SV ↓
     ↓
Increasing Schedule Risk
```

### Combined Deterioration

```text id="8hl8y6"
CPI ↓
SPI ↓
EAC ↑
TCPI ↑
     ↓
High-Risk Prediction
```

These scenarios can be used to validate whether the model responds logically to deteriorating project conditions.

---

# Model Validation Considerations

Because the objective is future-risk prediction, validation should account for the temporal structure of project data.

Randomly mixing future observations into training data can produce overly optimistic performance estimates.

A stronger validation strategy can use:

* Time-based train/test splitting
* Project-level splitting
* Grouped cross-validation
* Rolling-window validation

For example:

```text id="5l7u7d"
Earlier Milestones
       ↓
     Training
       ↓
Later Milestones
       ↓
      Test
```

This better represents the real-world use case where the model is trained on historical information and used to predict future project conditions.

---

# Future Improvements

## 1. Time-Series Modeling

Extend the framework using models designed for sequential project data:

* LSTM
* GRU
* Temporal Convolutional Networks

This could capture temporal dependencies between successive milestones.

---

## 2. Probabilistic Risk Prediction

Instead of producing only a binary classification, generate a probability:

```text
Predicted High-Risk Probability = 0.82
```

This allows project managers to prioritize projects based on risk severity.

---

## 3. Cost and Schedule Risk as Separate Targets

Develop separate models for:

```text
Cost Overrun Risk
       +
Schedule Overrun Risk
```

This can provide more granular information to project managers.

---

## 4. Multi-Class Risk Classification

Extend binary classification to:

```text
Low Risk
Moderate Risk
High Risk
Critical Risk
```

This can provide more actionable project risk categories.

---

## 5. Real-Time Project Monitoring

Integrate the predictive model with project management systems to continuously update:

* CPI
* SPI
* EAC
* TCPI
* Risk probability

and automatically flag deteriorating project phases.

---

## 6. Explainable Risk Dashboard

Develop an interactive dashboard showing:

```text
Project Health
     ↓
Current EVM Metrics
     ↓
Predicted Risk
     ↓
Risk Probability
     ↓
Top SHAP Drivers
     ↓
Recommended Investigation Area
```

Potential deployment technologies include Streamlit or a web-based dashboard.

---

## 7. Scenario and Sensitivity Analysis

Simulate hypothetical changes such as:

* Cost increases
* Schedule delays
* Productivity reductions
* Resource constraints
* Procurement delays

and evaluate their effect on predicted project risk.

---

# Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Earned Value Management
* Project Management Analytics
* Predictive Analytics
* Machine Learning
* Binary Classification
* Logistic Regression
* Random Forest
* XGBoost
* Feature Engineering
* Model Evaluation
* ROC-AUC
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Explainable AI
* SHAP
* Risk Prediction
* Early Warning Systems
* Infrastructure Project Analytics
* Cost Overrun Prediction
* Schedule Overrun Prediction
* Time-Aware Model Validation
* Data-Driven Decision Making

---

# Conclusion

The **EVM-Based Schedule/Cost Overrun Predictor** combines traditional project performance measurement with modern machine learning to create an early-warning framework for infrastructure project risks.

By engineering **CPI, SPI, CV, SV, EAC, and TCPI** from project milestone data and applying classification algorithms such as **Logistic Regression, Random Forest, and XGBoost**, the framework can identify patterns associated with future cost and schedule overruns.

The achieved **84% classification accuracy and 0.86 ROC-AUC** demonstrate the potential of the approach for distinguishing high-risk and low-risk project phases.

The addition of **SHAP-based explainability** makes the predictions more actionable by identifying the EVM indicators driving individual risk predictions.

Overall, the project demonstrates how **Earned Value Management, machine learning, and explainable AI can be combined to support proactive project monitoring and data-driven project management decisions**.
