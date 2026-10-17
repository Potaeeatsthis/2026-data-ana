import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("data/employee_churn_ecommerce_retail_5000_augmented.csv")

# Inspect the actual column names
print("Columns:", df.columns.tolist())

# 2. Create binary churn target from Termination Date
termination_col = "termination_date"

df["Churn"] = (
    df[termination_col]
    .notna()
    .astype(int)
)

# Check class distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# 3. Remove columns that would leak the answer
X = df.drop(columns=["Churn", termination_col, "employee_id"])
y = df["Churn"]

# 3. Split BEFORE applying the lottery
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining classes before lottery:")
print(y_train.value_counts())

# 4. Preprocess numerical and categorical features
numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 5. LOTTERY: randomly undersample the majority class
lottery = RandomUnderSampler(
    sampling_strategy=0.45,
    random_state=42
)

# 6. Build the model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("lottery", lottery),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate on the ORIGINAL, untouched test set

# Get the probability of churn (class 1)
churn_probabilities = model.predict_proba(X_test)[:, 1]

# Set the decision threshold
threshold = 0.25

# Predict churn if probability >= threshold
predictions = (churn_probabilities >= threshold).astype(int)

print(f"\nDecision threshold: {threshold}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

print("\n========== CONFUSION MATRIX ==========")
print("                 Predicted")
print("                 Not Churn   Churn")
print(f"Actual Not Churn    {tn:5d}     {fp:5d}")
print(f"Actual Churn       {fn:5d}     {tp:5d}")

print("\n========== CLASSIFICATION RESULTS ==========")
print(f"True Positives (TP):  {tp}")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

print("\n========== PERFORMANCE METRICS ==========")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not churned", "Churned"],
    zero_division=0
))

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

# 9. ROC curve (uses the probabilities, so it does not depend on the threshold)
fpr, tpr, roc_thresholds = roc_curve(y_test, churn_probabilities)
auc = roc_auc_score(y_test, churn_probabilities)
print("ROC AUC:", round(auc, 4))

# Where your chosen threshold sits on the curve
op_fpr = fp / (fp + tn)
op_tpr = tp / (tp + fn)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC = 0.5)")
plt.scatter(
    [op_fpr], [op_tpr],
    color="red", zorder=5,
    label=f"Threshold = {threshold} (TPR {op_tpr:.2f}, FPR {op_fpr:.2f})"
)
plt.xlabel("False Positive Rate (non-churners wrongly flagged)")
plt.ylabel("True Positive Rate / Recall (churners caught)")
plt.title("ROC Curve - Employee Churn")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
