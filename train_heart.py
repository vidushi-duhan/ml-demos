import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

FEATURES = ["age", "sex", "trestbps", "chol", "thalach", "exang", "fbs"]
TARGET = "num"

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv("data/heart.csv")

print("=" * 60)
print("INSPECTION REPORT")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"All columns: {list(df.columns)}")
print()

print("Feature value counts (present / missing):")
for feat in FEATURES:
    present = df[feat].notna().sum()
    missing = df[feat].isna().sum()
    print(f"  {feat:10s}  present={present}  missing={missing}")
print()

print(f"Target column '{TARGET}' distribution:")
for val, cnt in sorted(df[TARGET].value_counts().items()):
    print(f"  {val}: {cnt} rows")
print()

# ── 2. Prepare data ──────────────────────────────────────────────────────────
df_model = df[FEATURES + [TARGET]].copy()
df_model["target"] = (df_model[TARGET] > 0).astype(int)
df_model = df_model.drop(columns=[TARGET])

print("Binary class balance:")
print(df_model["target"].value_counts().sort_index().to_string())
print()

print("Median imputation per feature:")
total_imputed = 0
for feat in FEATURES:
    n_missing = df_model[feat].isna().sum()
    if n_missing > 0:
        median_val = df_model[feat].median()
        df_model[feat] = df_model[feat].fillna(median_val)
        print(f"  {feat}: imputed {n_missing} values with median={median_val:.2f}")
        total_imputed += n_missing
    else:
        print(f"  {feat}: 0 missing — no imputation needed")
print(f"  Total values imputed: {total_imputed}")
print()

# ── 3. Train ─────────────────────────────────────────────────────────────────
X = df_model[FEATURES]
y = df_model["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=1000)),
])

pipeline.fit(X_train, y_train)

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("EVALUATION")
print("=" * 60)
print(f"Test accuracy : {acc:.4f}")
print(f"Precision     : {prec:.4f}")
print(f"Recall        : {rec:.4f}")
print(f"ROC-AUC       : {auc:.4f}")
print(f"CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"                Pred 0   Pred 1")
print(f"  Actual 0  :  {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"  Actual 1  :  {cm[1,0]:6d}   {cm[1,1]:6d}")
print()

print("Feature ranges (full dataset):")
for feat in FEATURES:
    print(f"  {feat:10s}  min={df[feat].min():.1f}  max={df[feat].max():.1f}")
print()

print("Binary feature encodings:")
for feat in ["sex", "exang", "fbs"]:
    print(f"  {feat}: {dict(df[feat].value_counts().sort_index())}")
print()

# ── 5. Save model ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/heart_model.pkl")
print("Saved models/heart_model.pkl")
print()

# ── 6. Summary block ─────────────────────────────────────────────────────────
n_total = len(df_model)
n_disease = (y == 1).sum()
n_healthy = (y == 0).sum()

print("=" * 60)
print("=== SUMMARY FOR FRONTEND ===")
print(f"Total patients: {n_total}")
print(f"With disease (class 1): {n_disease}")
print(f"Without disease (class 0): {n_healthy}")
print(f"Missing values imputed: {total_imputed}")
print()
print(f"Test accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"CV accuracy (mean): {cv_scores.mean():.4f}")
print(f"CV accuracy (std): {cv_scores.std():.4f}")
print()
print("Slider ranges:")
print(f"Age: min={df['age'].min():.0f} max={df['age'].max():.0f}")
print(f"Resting BP: min={df['trestbps'].min():.0f} max={df['trestbps'].max():.0f}")
print(f"Cholesterol: min={df['chol'].min():.0f} max={df['chol'].max():.0f}")
print(f"Max heart rate: min={df['thalach'].min():.0f} max={df['thalach'].max():.0f}")
print()

# Decode binary features
sex_counts = df["sex"].value_counts().sort_index()
exang_counts = df["exang"].value_counts().sort_index()
fbs_counts = df["fbs"].value_counts().sort_index()

print("Binary encodings:")
print(f"Sex: 0={int(sex_counts.get(0,0))} patients, 1={int(sex_counts.get(1,0))} patients  (0=Female, 1=Male per UCI spec)")
print(f"Exercise-induced angina: 0={int(exang_counts.get(0,0))}, 1={int(exang_counts.get(1,0))}  (0=No, 1=Yes)")
print(f"Fasting blood sugar: 0={int(fbs_counts.get(0,0))}, 1={int(fbs_counts.get(1,0))}  (0=<=120mg/dl, 1=>120mg/dl)")
print("===========================")
