import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Usage: python make_augmented.py [input_csv]
# Default input is the original CSV in the same folder as this script.
HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "employee_churn_ecommerce_retail_5000_10cols.csv"
OUT = SRC.parent / "employee_churn_ecommerce_retail_5000_augmented.csv"
LOG = SRC.parent / "added_churners_log.csv"
TARGET_RATE = 0.25
rng = np.random.default_rng(42)

df = pd.read_csv(SRC)
y = df.termination_date.notna().astype(int)
feats = ["age","department","job_level","years_at_company","monthly_salary",
         "overtime_hours_per_month","job_satisfaction","performance_rating"]
num = [c for c in feats if c not in ("department","job_level")]
cat = ["department","job_level"]

# risk model learned from the REAL churners (employee_id deliberately excluded)
pre = ColumnTransformer([
  ("n", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), num),
  ("c", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]), cat)])
risk = Pipeline([("p",pre),("m",LogisticRegression(max_iter=1000))]).fit(df[feats], y)
p = risk.predict_proba(df[feats])[:,1]

# pick extra churners among active employees, weighted by risk (sampled, not just top-k)
active = np.where(y==0)[0]
n_add = int(round(TARGET_RATE*len(df))) - int(y.sum())
w = p[active]**2; w /= w.sum()
chosen = rng.choice(active, size=n_add, replace=False, p=w)

# dates drawn from the real termination-date distribution
real_dates = df.termination_date.dropna().values
new = df.copy()
new.loc[new.index[chosen], "termination_date"] = rng.choice(real_dates, size=n_add)
new.to_csv(OUT, index=False)
df.iloc[chosen][["employee_id"]].assign(assigned_date=new.loc[new.index[chosen],"termination_date"].values).to_csv(LOG, index=False)

z = new.termination_date.notna().astype(int)
print("added:", n_add, "| churn rate:", round(y.mean(),3), "->", round(z.mean(),3))
for c in num: print(f"{c:28s} corr before {np.corrcoef(y,df[c].fillna(df[c].median()))[0,1]:+.3f}  after {np.corrcoef(z,df[c].fillna(df[c].median()))[0,1]:+.3f}")
