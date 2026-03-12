# Bree Take-Home: ML Loan Default Prediction

## How to Run

1. Clone this repo
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Open `loan_ml_analysis.ipynb` in Jupyter
4. Run all cells from top to bottom

The dataset `loan_applications.csv` is already included — no need to regenerate it.

## Approach & Key Decisions

- **Model:** Gradient Boosting Classifier (sklearn)
  - Chosen for best accuracy on tabular data, native feature importances, and sample_weight support for class imbalance
- **Class imbalance:** Upweighted defaulted samples by 2.4x via sample_weight
- **Missing values:** Treated `doc_income_missing` as a binary feature — missingness is a strong risk signal (49% vs 26% default rate)
- **Ongoing applications:** Excluded from training — no known outcome. Noted as a selection bias risk
- **rule_based_score kept as feature:** Functions as model stacking rather than leakage — the score doesn't cleanly separate defaulted from repaid (confirmed by EDA), so the model cannot simply copy it
- **Threshold tuning:** Optimized at 0.57 by maximizing F1 on training set, not default 0.5
- **Explainability:** Feature importances + permutation importance + per-applicant human-readable explanation function. In production, would add SHAP values as the industry standard for regulated lending environments

## Results vs Baseline

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| AUC-ROC | 0.722 | 0.720 |
| Recall (default) | 0.33 | 0.56 |
| F1 (default) | 0.43 | 0.53 |
| Defaults caught | 36 | 61 |
| Good applicants wrongly denied | 22 | 59 |

AUC is essentially tied. The ML model's advantage is catching 25 more defaults (+69%) at the cost of 37 more wrongful denials (+14%).

## Fairness Finding

- Employed vs self-employed default rates are statistically indistinguishable (p=0.904)
- Rule-based model has a 21.5pp approval gap between them — proxy discrimination
- ML model reduces this gap to 16.4pp by learning from actual outcomes
- Unemployed penalization IS justified (56% default rate, p=0.000)
- Recommendation: keep employment_status with quarterly fairness audits

## What I'd Do With More Time

1. Add SHAP values for rigorous per-applicant explanations
2. Hyperparameter tuning via Optuna
3. Fairness-constrained training with fairlearn (equalized odds)
4. Model monitoring dashboard tracking PSI + fairness metrics
5. Backtest on resolved ongoing applications

## Loom Video
https://www.loom.com/share/129e0a5770664e9e8cad61842ce90dc7
