"""Find high-risk employee segments (e.g. bottom-20% satisfaction AND top-20% overtime)
and test whether using them as features helps the model.
Usage: python segment_analysis.py [csv_file]"""
import sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

path = sys.argv[1] if len(sys.argv) > 1 else "employee_churn_ecommerce_retail_5000_augmented.csv"
df = pd.read_csv(path)
y = df["termination_date"].notna().astype(int)
base = y.mean()
print(f"File: {path} | employees: {len(df)} | overall churn rate: {base:.1%}\n")

# ---- 1. Risk flags built from percentiles (bottom/top 20%) ----
flags = pd.DataFrame({
    "low_satisfaction (bottom 20%)":  df.job_satisfaction <= df.job_satisfaction.quantile(.20),
    "high_overtime (top 20%)":        df.overtime_hours_per_month >= df.overtime_hours_per_month.quantile(.80),
    "low_salary (bottom 20%)":        df.monthly_salary <= df.monthly_salary.quantile(.20),
    "low_performance (bottom 20%)":   df.performance_rating <= df.performance_rating.quantile(.20),
    "short_tenure (bottom 20%)":      df.years_at_company <= df.years_at_company.quantile(.20),
    "entry_level":                    df.job_level == "Entry",
}).fillna(False)

def seg(mask, name):
    n = int(mask.sum()); r = y[mask].mean() if n else np.nan
    return {"segment": name, "employees": n, "churn_rate": f"{r:.1%}", "lift_vs_avg": f"{r/base:.1f}x"}

rows = [seg(flags[c], c) for c in flags.columns]
a, b = flags.iloc[:, 0], flags.iloc[:, 1]
rows += [seg(a & b, "low_satisfaction AND high_overtime"),
         seg(a & b & flags["entry_level"], "  ... AND entry_level"),
         seg(~flags.any(axis=1), "no risk flags at all")]
print("SEGMENT CHURN RATES"); print(pd.DataFrame(rows).to_string(index=False))

flags["n_flags"] = flags.iloc[:, :6].sum(axis=1)
tbl = y.groupby(flags["n_flags"]).agg(employees="count", churn_rate="mean")
print("\nCHURN RATE BY NUMBER OF RISK FLAGS (0-6)")
print(tbl.assign(churn_rate=(tbl.churn_rate*100).round(1).astype(str)+"%").to_string())

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(tbl.index.astype(str), tbl.churn_rate*100, color="#3b6ea5")
ax.axhline(base*100, color="red", linestyle="--", label=f"average churn {base:.1%}")
for i, (n, r) in enumerate(zip(tbl.employees, tbl.churn_rate)):
    ax.text(i, r*100+1, f"{r:.0%}\n(n={n})", ha="center", fontsize=8)
ax.set_xlabel("Number of risk flags an employee has"); ax.set_ylabel("Churn rate (%)")
ax.set_title("More risk flags -> higher churn"); ax.legend(); plt.tight_layout()
plt.savefig("segment_churn.png", dpi=150)

# ---- 2. Do the flags help the model? (Logistic + lottery, 5-fold) ----
num = ['age','years_at_company','monthly_salary','overtime_hours_per_month','job_satisfaction','performance_rating']
cat = ['department','job_level']
def fpr_at(yy, p, r):
    f, t, _ = roc_curve(yy, p); return f[np.searchsorted(t, r)]
def cv_eval(X, cols_num, cols_cat):
    pre = ColumnTransformer([("n", SkPipe([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols_num),
                             ("c", SkPipe([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cols_cat)])
    m = Pipeline([("p", pre), ("lottery", RandomUnderSampler(sampling_strategy=.45, random_state=42)),
                  ("c", LogisticRegression(max_iter=2000))])
    p = np.zeros(len(y)); yv = y.values
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, yv):
        p[te] = m.fit(X.iloc[tr], yv[tr]).predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(yv, p), average_precision_score(yv, p), 1000*fpr_at(yv, p, .6)
X0 = df[num + cat].copy()
X1 = X0.copy(); fl = flags.iloc[:, :6].astype(float); fl.columns = [f"flag{i}" for i in range(6)]
X1 = pd.concat([X1, fl], axis=1); X1["both_sat_ot"] = (a & b).astype(float); X1["n_flags"] = flags["n_flags"].astype(float)
extra = list(fl.columns) + ["both_sat_ot", "n_flags"]
print("\nDO THE SEGMENT FLAGS HELP THE MODEL? (logistic + lottery, 5-fold)")
r0 = cv_eval(X0, num, cat); r1 = cv_eval(X1, num + extra, cat)
print(f"{'':28s} {'AUC':>6s} {'PR-AUC':>7s} {'false alarms/1000 @60% recall':>32s}")
print(f"{'original columns only':28s} {r0[0]:6.3f} {r0[1]:7.3f} {r0[2]:32.0f}")
print(f"{'+ segment flags':28s} {r1[0]:6.3f} {r1[1]:7.3f} {r1[2]:32.0f}")
