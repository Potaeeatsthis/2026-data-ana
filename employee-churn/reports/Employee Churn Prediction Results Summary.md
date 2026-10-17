# Employee Churn Prediction: Results Summary

Predictive analytics project on 5,000 e-commerce and retail employees, with a prescriptive extension.

## 1. Project at a glance

| Item | Result | Simple meaning |
| --- | --- | --- |
| Goal | Predict who will leave | Find at-risk employees early |
| Data | 5,000 employees, 10 columns | Small dataset, limited information |
| Leavers | 12.3% (613 people) | Very imbalanced: about 1 in 8 leaves |
| Test group | 1,000 people, 123 real leavers | The "exam" the models never saw |
| Models compared | Logistic Regression, Random Forest, XGBoost, Neural Network | Simple, tree-based and deep learning |

## 2. What each score means

| Score | Simple meaning | Good is |
| --- | --- | --- |
| Recall | Of the real leavers, how many we caught | High |
| Precision | Of the people flagged, how many really left | High |
| F1 | One score blending recall and precision | High |
| ROC-AUC | How often a leaver is ranked riskier than a stayer (0.5 = coin flip) | High |
| PR-AUC | How well we find leavers (random = 0.12) | High |
| Brier | How honest the percentages are | Low |

## 3. Model comparison (test set)

| Model | ROC-AUC | PR-AUC | Precision | Recall | Plain-English description |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.736 | 0.285 | 25% | 50% | Best at ranking, easy to explain |
| XGBoost | 0.719 | 0.281 | 24% | 49% | Same performance, harder to explain |
| Random Forest | 0.718 | 0.275 | 26% | 43% | Most careful, misses the most leavers |
| Neural Network | 0.722 | 0.270 | 24% | 59% | Catches the most leavers, flags the most people |

All four are close. The differences are inside the margin of error, so none is clearly better.

## 4. Why I recommend each one

| Model | Decision | Reason |
| --- | --- | --- |
| Logistic Regression | **Use** | Top scores, simple, and you can say exactly why someone is flagged |
| XGBoost | Backup | Equal performance, useful if you add more data later |
| Random Forest | Skip | Lowest scores, no advantage |
| Neural Network | Skip | Too little data, and hard to explain |

## 5. What drives people to leave

| Factor | Finding | Meaning |
| --- | --- | --- |
| Overtime | Over 20 hours a month: 39% leave, under 4 hours: 7% | Heavy overtime is the strongest warning sign |
| Job satisfaction | Score under 40: 34% leave | Unhappy staff leave more |
| Job level | Entry level 20%, managers 2% | Junior staff are the most at risk |
| Department | Fulfillment highest at 19% | Some teams have bigger problems |

## 6. How well the model separates risk (test set)

| Risk group | Actual leave rate | Meaning |
| --- | --- | --- |
| Top 10% highest risk | 31% | Almost 3x the average |
| Next 20% | 21% | Elevated risk |
| Remaining 70% | 7% | Low risk |
| Top 20% overall | Contains 45% of all leavers | Random picking would find only 20% |
