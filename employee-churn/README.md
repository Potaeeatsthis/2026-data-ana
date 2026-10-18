# Employee Churn Prediction (E-commerce / Retail)

Predicts which employees are likely to leave. The data is heavily imbalanced (12.3% churn), so training uses a "lottery": random undersampling of stayers. Four models were built and compared step by step, each in its own notebook.

## Data

| File | What it is |
|---|---|
| `data/employee_churn_ecommerce_retail_5000_10cols.csv` | 5,000 employees, 10 columns. A blank `termination_date` means the employee is still employed (churn = 0). 613 of 5,000 left (12.3%). |

## Notebooks (read in order)

| Notebook | What it covers |
|---|---|
| `notebooks/01_random_forest_lottery.ipynb` | **Model 1** - random forest + lottery, fixed threshold 0.45. Shows its weaknesses (overfitting, meaningless threshold). |
| `notebooks/02_logistic_regression_lottery.ipynb` | **Model 2 (final)** - logistic regression + lottery, threshold chosen from a recall target. Includes risk tiers and feature weights. |
| `notebooks/03_lottery_ticket_hypothesis.ipynb` | **Models 3 and 4** - a small neural network (dense, 3,360 weights) and its pruned "winning ticket" (190 weights, 5.7%). Tests whether the winning ticket beats a randomly re-initialised copy over 10 seeds. |
| `notebooks/04_compare_models_confusion_matrices.ipynb` | All four models side by side: confusion matrices, ROC curves, and false alarms at the same recall. |

## The four models

| # | Model | Decision rule |
|---|---|---|
| 1 | Random forest + lottery | fixed threshold 0.45 |
| 2 | Logistic regression + lottery | threshold from a 60% recall target |
| 3 | Neural network, dense (all 3,360 weights) + lottery | threshold from a 60% recall target |
| 4 | Neural network, winning ticket (190 weights, 5.7%) + lottery | threshold from a 60% recall target |

All four use the same data, the same 80/20 split, the same blank-filling and the same lottery.

## Results (original data, test set: 123 leavers + 877 stayers)

| | 1 Random forest | 2 Logistic regression | 3 Neural net (dense) | 4 Winning ticket |
|---|---|---|---|---|
| Leavers caught | 54 (43.9%) | 77 (62.6%) | 68 (55.3%) | 76 (61.8%) |
| False alarms | 163 | 251 | 184 | 212 |
| Precision | 24.9% | 23.5% | 27.0% | 26.4% |
| Balanced accuracy | 0.627 | 0.670 | 0.672 | 0.688 |
| ROC AUC | 0.699 | 0.738 | 0.745 | 0.746 |

The models catch different numbers of leavers, so compare false alarms **at the same recall**:

| Recall | 1 Random forest | 2 Logistic | 3 Dense net | 4 Winning ticket |
|---|---|---|---|---|
| 50% | 194 | 167 | 153 | 154 |
| 60% | 289 | 209 | 217 | 200 |
| 70% | 373 | 305 | 315 | 300 |
| 80% | 450 | 402 | 418 | 419 |

## Key findings

- **The random forest is the weakest.** It overfits (AUC 0.988 on training data vs 0.699 on test) and raises the most false alarms at every recall level. The other three models each raise about 17-19% fewer false alarms on average.
- **Logistic regression and the neural networks are essentially tied** (AUC about 0.74-0.75). Differences at this level are noise on a single split.
- **The winning ticket matches the dense network with about 94% of the weights removed.** However, a random re-initialisation of the same sparse structure did almost as well (about +0.001-0.003 AUC difference over 10 seeds), so there is little evidence of a distinct winning-ticket effect on this data.
- **Logistic regression stays the final model.** The neural network does not beat it, and logistic regression is simpler and explainable.
- **The ceiling is the data.** About ten model types all reached an AUC of about 0.75, so new features (manager, promotions, time since last raise) would help more than a different algorithm.
- Plain accuracy is misleading here ("flag nobody" scores 87.7%). Use balanced accuracy, ROC AUC and PR-AUC.

## Two different "lotteries"

- **The data lottery** (used by every model): random undersampling of stayers in the training set, to balance the classes.
- **The Lottery Ticket Hypothesis** (models 3 and 4 only): a claim about neural networks, that a large network contains a small sub-network that trains as well as the full one. Found by training, pruning the smallest weights, and rewinding the survivors to their original values.

They are unrelated ideas that share a name.

## How the models work

1. Label: blank `termination_date` = stayed, a date = left. `employee_id` and `termination_date` are not used as inputs.
2. Split 80/20 (stratified) before anything else. Blanks are filled with the median (numbers) or most frequent value (categories), learned from training data only.
3. Lottery: stayers are randomly undersampled in the training set only (`sampling_strategy=0.45`).
4. Model: random forest, logistic regression, or the neural network (dense or pruned).
5. Threshold: for models 2-4, chosen from 5-fold cross-validated training scores to hit a recall target (default 60%). The test set is never used for tuning.
6. Evaluation on the untouched test set: confusion matrix, balanced accuracy, ROC AUC, PR-AUC.

## Limitations

- Precision stays low (about a quarter of flags are real leavers, against 12.3% if flagging at random) because the signal in the columns is moderate.
- One 80/20 split is noisy; cross-validation (in notebooks 01 and 02) agrees on the direction.
- The neural network numbers in notebook 04 come from a single random start; notebook 03 shows the seed-to-seed spread is about 0.002 AUC.
- The neural network is a small NumPy implementation with layer-wise magnitude pruning, not a deep-learning framework.

## Run

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

Each notebook looks for the CSV in `data/`, so run them from the `notebooks/` folder (or open them in VS Code). The neural network needs only NumPy, with no deep-learning library.
