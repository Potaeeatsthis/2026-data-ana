# Employee Churn Prediction (E-commerce / Retail)

Predicts which employees are likely to leave. The data is heavily imbalanced (12.3% churn), so training uses a "lottery": random undersampling of stayers. The notebooks show how the model changed step by step.

## Data

| File | What it is |
|---|---|
| `data/employee_churn_ecommerce_retail_5000_10cols.csv` | 5,000 employees, 10 columns. A blank `termination_date` means the employee is still employed (churn = 0). |

## Notebooks (read in order)

| Notebook | What it covers |
|---|---|
| `notebooks/01_random_forest_lottery.ipynb` | **Model 1** - random forest + lottery, fixed threshold 0.45. Shows its weaknesses (overfitting, meaningless threshold). |
| `notebooks/02_logistic_regression_lottery.ipynb` | **Model 2 (final)** - logistic regression + lottery, threshold chosen from a recall target. Includes risk tiers and feature weights. |
| `notebooks/03_lottery_ticket_hypothesis.ipynb` | Separate experiment: does a pruned neural network (a "winning ticket") match the full network on this data? |
| `notebooks/04_compare_models_confusion_matrices.ipynb` | Confusion matrices and ROC curves for the models side by side. |

## Results (original data, test set: 123 leavers + 877 stayers)

| | Model 1: random forest | Model 2: logistic regression |
|---|---|---|
| Decision rule | fixed threshold 0.45 | threshold from a 60% recall target |
| Leavers caught | 54 (43.9%) | 77 (62.6%) |
| False alarms | 163 | 251 |
| Precision | 24.9% | 23.5% (random flagging: 12.3%) |
| Balanced accuracy | 0.627 | **0.670** |
| ROC AUC (test / 5-fold CV) | 0.699 / 0.719 | **0.738 / 0.747** |

- The two models catch different numbers of leavers, so compare false alarms **at the same recall**: Model 2 raises about 18% fewer false alarms on average (see notebook 04).
- Plain accuracy is misleading here ("flag nobody" scores 87.7%). Use balanced accuracy, ROC AUC and PR-AUC.
- About ten model types all reached an AUC of about 0.75, so the limit is the information in the columns. New features (manager, promotions, time since last raise) would help more than a different algorithm.
- Lottery ticket experiment: a small network keeps its AUC (about 0.744) with about 96% of weights removed, but a random re-initialisation of the same sparse structure does almost as well, so there is little evidence of a distinct winning-ticket effect on this data.

## How the final model works

1. Label: blank `termination_date` = stayed, a date = left. `employee_id` and `termination_date` are not used as inputs.
2. Split 80/20 (stratified) before anything else. Blanks are filled with the median (numbers) or most frequent value (categories), learned from training data only.
3. Lottery: stayers are randomly undersampled in the training set only (`sampling_strategy=0.45`).
4. Logistic regression.
5. Threshold: chosen from 5-fold cross-validated training scores to hit a recall target (default 60%).
6. Evaluation on the untouched test set: confusion matrix, balanced accuracy, ROC AUC, PR-AUC, risk tiers.

## Run

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

Each notebook looks for the CSV in `data/`, so run them from the `notebooks/` folder (or open them in VS Code).
