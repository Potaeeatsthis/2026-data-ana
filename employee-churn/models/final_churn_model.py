"""Recommended setup for the ORIGINAL churn data (lottery kept).
Usage: python final_churn_model.py [csv_file]      (default: original CSV in this folder)"""
import sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, balanced_accuracy_score,
                             roc_auc_score, average_precision_score, roc_curve)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

RECALL_TARGET = 0.60

path = sys.argv[1] if len(sys.argv) > 1 else "employee_churn_ecommerce_retail_5000_10cols.csv"
df = pd.read_csv(path)
y = df["termination_date"].notna().astype(int)
X = df.drop(columns=["termination_date", "employee_id"])          # employee_id is just noise
print(f"File: {path} | churn rate: {y.mean():.1%} ({y.sum()} of {len(y)})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
num = X_train.select_dtypes(include="number").columns.tolist()
cat = X_train.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("numeric", SklearnPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("categorical", SklearnPipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat)])
model = Pipeline([("preprocessor", pre),
                  ("lottery", RandomUnderSampler(sampling_strategy=0.45, random_state=42)),   # lottery kept
                  ("classifier", LogisticRegression(max_iter=1000))])

# Threshold chosen from TRAINING data only, to hit the recall target (test set is never used for tuning)
oof = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba")[:, 1]
pos = np.sort(oof[y_train.values == 1])
threshold = pos[int(np.floor((1 - RECALL_TARGET) * len(pos)))]

model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
pred = (p >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

P, N = int(y_test.sum()), int((y_test == 0).sum())
print(f"\nTest set: {len(y_test)} employees = {P} leavers + {N} stayers")
print(f"Recall target {RECALL_TARGET:.0%} -> decision threshold {threshold:.3f} (chosen from training data)")
print("\n========== CONFUSION MATRIX ==========")
print("                 Predicted")
print("                 Not Churn   Churn")
print(f"Actual Not Churn    {tn:5d}     {fp:5d}")
print(f"Actual Churn       {fn:5d}     {tp:5d}")
print("\n========== CLASSIFICATION RESULTS ==========")
print(f"True Positives (TP):  {tp:4d}   leavers correctly flagged        ({tp/(tp+fn):.1%} of all leavers)")
print(f"True Negatives (TN):  {tn:4d}   stayers correctly left alone     ({tn/(tn+fp):.1%} of all stayers)")
print(f"False Positives (FP): {fp:4d}   stayers wrongly flagged (false alarms) ({fp/(fp+tn):.1%} of all stayers)")
print(f"False Negatives (FN): {fn:4d}   leavers we missed               ({fn/(fn+tp):.1%} of all leavers)")
print("\n========== PERFORMANCE ==========")
print(classification_report(y_test, pred, labels=[0, 1], target_names=["Not churned", "Churned"], zero_division=0))
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.3f}")
print(f"ROC AUC:           {roc_auc_score(y_test, p):.3f}   (0.5 = guessing)")
print(f"PR-AUC:            {average_precision_score(y_test, p):.3f}   (guessing = {y_test.mean():.3f})")
print(f"\nYardsticks: 'flag nobody' accuracy = {N/len(y_test):.1%} | random flagging precision = {y_test.mean():.1%} "
      f"| this model's precision = {tp/max(tp+fp,1):.1%}")

# Stability check: full pipeline, 5-fold CV on all data
aucs, aps = [], []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
    pp = model.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    aucs.append(roc_auc_score(y.iloc[te], pp)); aps.append(average_precision_score(y.iloc[te], pp))
print(f"\n5-fold CV: AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | PR-AUC {np.mean(aps):.3f} +/- {np.std(aps):.3f}")
model.fit(X_train, y_train)

# Risk tiers by rank (robust to the lottery inflating probabilities)
t = pd.DataFrame({"p": p, "y": y_test.values})
t["rank_pct"] = t.p.rank(pct=True, ascending=False)
t["tier"] = pd.cut(t.rank_pct, [0, .10, .30, 1.0], labels=["HIGH (top 10%)", "MEDIUM (next 20%)", "LOW (bottom 70%)"])
tiers = t.groupby("tier", observed=True).y.agg(employees="count", leavers="sum", churn_rate="mean")
tiers["share_of_all_leavers"] = tiers.leavers / P
print("\n========== RISK TIERS (test set) ==========")
print(tiers.assign(churn_rate=(tiers.churn_rate*100).round(1).astype(str)+"%",
                   share_of_all_leavers=(tiers.share_of_all_leavers*100).round(1).astype(str)+"%").to_string())

f, tp_r, _ = roc_curve(y_test, p)
plt.figure(figsize=(7, 6))
plt.plot(f, tp_r, linewidth=2, label=f"Logistic + lottery (AUC = {roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
plt.scatter([fp/(fp+tn)], [tp/(tp+fn)], color="red", zorder=5, label=f"Chosen threshold {threshold:.2f}")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (recall)")
plt.title("ROC - original data"); plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("roc_final.png", dpi=150); plt.show()