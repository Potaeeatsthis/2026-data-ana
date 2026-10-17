"""Confusion matrix pictures for ALL models (old + current), individually and side by side.
Usage: python confusion_matrices_all_models.py [csv_file]
Default csv: the original data in the current folder.

To add another model: add one entry to the MODELS list near the bottom.
  threshold = a number (fixed cut-off)   OR   ("recall", 0.60) to pick the cut-off from a recall target."""
import re, sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

path = sys.argv[1] if len(sys.argv) > 1 else "employee_churn_ecommerce_retail_5000_10cols.csv"
df = pd.read_csv(path)
y = df["termination_date"].notna().astype(int)
X = df.drop(columns=["termination_date", "employee_id"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
num = X_train.select_dtypes(include="number").columns.tolist()
cat = X_train.select_dtypes(exclude="number").columns.tolist()
print(f"File: {path} | churn rate {y.mean():.1%} | test set: {int(y_test.sum())} leavers + {int((y_test==0).sum())} stayers\n")

def make_pipeline(classifier):
    pre = ColumnTransformer([
        ("numeric", SklearnPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("categorical", SklearnPipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                         ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("preprocessor", pre),
                     ("lottery", RandomUnderSampler(sampling_strategy=0.45, random_state=42)),   # lottery in every model
                     ("classifier", classifier)])

def pick_threshold(model, spec):
    if not isinstance(spec, tuple):
        return float(spec)
    target = spec[1]     # threshold from TRAINING data only, to hit the recall target
    oof = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            method="predict_proba")[:, 1]
    pos = np.sort(oof[y_train.values == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])

def draw_matrix(ax, tn, fp, fn, tp, title):
    cm = np.array([[tn, fp], [fn, tp]])
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    names = [["TN", "FP"], ["FN", "TP"]]
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}\n{row_pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if row_pct[i, j] > 0.5 else "black", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not churn", "Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)

# ---------------- the models ----------------
MODELS = [
    ("Old: Random forest + lottery",          RandomForestClassifier(n_estimators=200, random_state=42), 0.45),
    ("Current: Logistic regression + lottery", LogisticRegression(max_iter=1000),                          ("recall", 0.60)),
    # Example to add more:
    # ("Logistic @ 80% recall", LogisticRegression(max_iter=1000), ("recall", 0.80)),
]

results = []
for name, clf, spec in MODELS:
    model = make_pipeline(clf)
    thr = pick_threshold(model, spec)
    model.fit(X_train, y_train)
    pred = (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rec, prec = tp / (tp + fn), tp / max(tp + fp, 1)
    results.append((name, thr, tn, fp, fn, tp, rec, prec))
    title = f"{name}\nthreshold {thr:.2f} | recall {rec:.0%} | precision {prec:.0%}"
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); draw_matrix(ax, tn, fp, fn, tp, title); fig.tight_layout()
    fig.savefig(f"confusion_{re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')}.png", dpi=150)

print(f"{'model':42s} {'thr':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>5s} {'recall':>7s} {'precision':>9s}")
for name, thr, tn, fp, fn, tp, rec, prec in results:
    print(f"{name:42s} {thr:5.2f} {tp:4d} {fp:4d} {fn:4d} {tn:5d} {rec:7.1%} {prec:9.1%}")

# side-by-side picture
fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5.8), squeeze=False)
for ax, (name, thr, tn, fp, fn, tp, rec, prec) in zip(axes[0], results):
    draw_matrix(ax, tn, fp, fn, tp, f"{name}\nthreshold {thr:.2f} | recall {rec:.0%} | precision {prec:.0%}")
fig.suptitle("Confusion matrices - all models (% = share of each actual group; dark = high share)", fontsize=12)
fig.tight_layout()
fig.savefig("confusion_matrices_all_models.png", dpi=150)
plt.show()
