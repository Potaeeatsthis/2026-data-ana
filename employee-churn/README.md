# Employee Churn Prediction (E-commerce / Retail)

Predicts which employees are likely to leave, using a random-undersampling "lottery" to handle class imbalance (12.3% churn).

## Project structure

```
data/         original dataset + synthetic (augmented) experiment data
models/       final model (models/) and earlier versions (models/old/)
scripts/      data augmentation and segment analysis
notebooks/    exploratory notebook
reports/      written results summary
figures/      ROC curves and segment chart
```

## Data

| File | What it is |
|---|---|
| `data/employee_churn_ecommerce_retail_5000_10cols.csv` | **Original data** - 5,000 employees, 12.3% churn. Use this for real conclusions. |
| `data/employee_churn_ecommerce_retail_5000_augmented.csv` | **Synthetic experiment** - 637 active employees were given a termination date (churn 25%), chosen to follow patterns in the real data. Results on this file are inflated. |
| `data/added_churners_log.csv` | Which employees were changed in the augmented file. |

## Model versions

| Version | File | Setup | Data | Balanced acc. | ROC AUC |
|---|---|---|---|---|---|
| v1 (old) | `models/old/v1_rf_lottery.py` | Random forest + lottery, fixed threshold | Augmented (25%) | 0.734 | 0.829 |
| v1 (old) | same, reproduced | Random forest + lottery, threshold 0.45 | Original (12.3%) | 0.627 | 0.699 |
| **Final** | `models/final_churn_model.py` | Logistic regression + lottery, threshold from 60% recall target | Original (12.3%) | **0.670** | **0.738** |

Final model on the original test set: catches 77 of 123 leavers (62.6%), 251 false alarms, precision 23.5% (random flagging would be 12.3%). 5-fold CV AUC 0.747 +/- 0.016.

Ten model families (logistic, LDA, random forest, extra trees, gradient boosting, SVM, neural net, blends...) all reached an AUC of about 0.75 on the original data, so the limit comes from the information in the columns, not the algorithm.

## How the final model works

1. Label: blank `termination_date` = stayed, a date = left. `employee_id` and `termination_date` are not used as inputs.
2. Split 80/20, stratified. Blanks filled with median (numeric) / most frequent (categorical), learned on training data only.
3. Lottery: stayers randomly undersampled in the training set (`sampling_strategy=0.45`).
4. Logistic regression.
5. Decision threshold chosen from cross-validated training scores to hit a recall target (`RECALL_TARGET`, default 0.60).
6. Evaluation on the untouched test set: confusion matrix, balanced accuracy, ROC AUC, PR-AUC, risk tiers.

## Run (from the repository root)

```powershell
pip install -r requirements.txt

# final model (original data)
python models/final_churn_model.py data/employee_churn_ecommerce_retail_5000_10cols.csv

# high-risk segment analysis
python scripts/segment_analysis.py data/employee_churn_ecommerce_retail_5000_10cols.csv

# regenerate the synthetic data (optional): args = input csv, target churn rate, strength
python scripts/make_augmented.py data/employee_churn_ecommerce_retail_5000_10cols.csv 0.25

# old v1 model (reads the augmented csv)
python models/old/v1_rf_lottery.py
```

Plots are saved to the folder you run from; move them to `figures/` if you want to keep them.

## Notes

- The augmented data is synthetic and was built to make the signal easier to learn. Report it as an experiment, never as evidence of real-world performance.
- Plain accuracy is misleading here ("flag nobody" scores 87.7%). Use balanced accuracy, ROC AUC and PR-AUC.
