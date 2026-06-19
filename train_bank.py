import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ── Config ───────────────────────────────────────────────────────────────────
NUMERIC_FEATURES  = ["age"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "housing", "loan", "poutcome"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "y"
DROP_COLS = ["duration"]   # target leakage

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv("data/bank.csv")

print("=" * 60)
print("INSPECTION REPORT")
print("=" * 60)
print(f"Total rows   : {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()

# Target
vc = df[TARGET].value_counts()
total = len(df)
print("Target distribution:")
for val, cnt in vc.items():
    print(f"  '{val}': {cnt:6d}  ({cnt/total*100:.2f}%)")
print()

# poutcome check
print("poutcome unique values and counts:")
print(df["poutcome"].value_counts(dropna=False).to_string())
print()

# Duration confirm drop
print(f"Dropping 'duration' (target leakage): present={('duration' in df.columns)}")
print()

# Missing / unknown per candidate feature
print("Missing / 'unknown' per candidate feature:")
for feat in ALL_FEATURES:
    nan_n = df[feat].isna().sum()
    unk_n = (df[feat] == "unknown").sum() if not pd.api.types.is_numeric_dtype(df[feat]) else 0
    print(f"  {feat:<15}: NaN={nan_n}  unknown={unk_n}")
print()

# ── 2. Prepare data ──────────────────────────────────────────────────────────
df_model = df[ALL_FEATURES + [TARGET]].copy()

# Drop duration (already not in ALL_FEATURES, but safety check)
for col in DROP_COLS:
    if col in df_model.columns:
        df_model.drop(columns=[col], inplace=True)

# Encode target: yes=1, no=0
df_model[TARGET] = (df_model[TARGET] == "yes").astype(int)

print("Binary class balance after encoding:")
bc = df_model[TARGET].value_counts().sort_index()
print(f"  0 (no) : {bc[0]:6d}  ({bc[0]/len(df_model)*100:.2f}%)")
print(f"  1 (yes): {bc[1]:6d}  ({bc[1]/len(df_model)*100:.2f}%)")
print()

# Treat 'unknown' as NaN so imputer handles it
for feat in CATEGORICAL_FEATURES:
    df_model[feat] = df_model[feat].replace("unknown", np.nan)

print("'unknown' replaced with NaN for imputation.")
print()

# ── 3. Build pipeline ────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,  NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )),
])

# ── 4. Train / test split ────────────────────────────────────────────────────
X = df_model[ALL_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}")
print(f"Train positives: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"Test  positives: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
print()

pipeline.fit(X_train, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_prob)
cm    = confusion_matrix(y_test, y_pred)

cv = cross_val_score(
    pipeline, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
)

print("=" * 60)
print("EVALUATION")
print("=" * 60)
print(f"Test accuracy  : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"ROC-AUC        : {auc:.4f}")
print(f"CV ROC-AUC     : {cv.mean():.4f} +/- {cv.std():.4f}")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"                Pred 0   Pred 1")
print(f"  Actual 0  :  {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"  Actual 1  :  {cm[1,0]:6d}   {cm[1,1]:6d}")
print()
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

# Feature names after OHE
ohe_cats = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
)
feature_names = NUMERIC_FEATURES + list(ohe_cats)
coefs = pipeline.named_steps["clf"].coef_[0]
coef_df = (
    pd.Series(coefs, index=feature_names)
    .abs()
    .sort_values(ascending=False)
)
print("Top 15 feature importances (abs coefficient):")
for feat, val in coef_df.head(15).items():
    bar = "#" * int(val * 5)
    print(f"  {feat:<35} {val:.4f}  {bar}")
print()

# ── 6. Subscription rate by poutcome (the big signal) ────────────────────────
print("Subscription rate by poutcome (in original data):")
df["_sub"] = (df[TARGET] == "yes").astype(int)
grp = (
    df.groupby("poutcome", dropna=False)["_sub"]
    .agg(count="count", subscribed="sum")
    .assign(rate=lambda x: (x["subscribed"] / x["count"] * 100).round(1))
    .sort_values("rate", ascending=False)
)
for cat, row in grp.iterrows():
    print(f"  {str(cat):<15}: {row['rate']:5.1f}%  (n={int(row['count'])})")
df.drop(columns=["_sub"], inplace=True)
print()

# ── 7. Save model ────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/bank_model.pkl")
print("Saved models/bank_model.pkl")
print()

# ── 8. Summary block ─────────────────────────────────────────────────────────
print("=" * 60)
print("=== SUMMARY FOR FRONTEND ===")
print(f"Total records  : {len(df):,}")
yes_n = int(vc.get("yes", 0))
no_n  = int(vc.get("no",  0))
print(f"Subscribed yes : {yes_n:,}  ({yes_n/total*100:.2f}%)")
print(f"Subscribed no  : {no_n:,}  ({no_n/total*100:.2f}%)")
print(f"Imbalance      : {no_n/yes_n:.1f}x")
print()
print(f"Test accuracy  : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"ROC-AUC        : {auc:.4f}")
print(f"CV ROC-AUC     : {cv.mean():.4f} +/- {cv.std():.4f}")
print()
print("Confusion matrix:")
print(f"  True Negatives  (no  -> no) : {cm[0,0]:,}")
print(f"  False Positives (no  -> yes): {cm[0,1]:,}")
print(f"  False Negatives (yes -> no) : {cm[1,0]:,}")
print(f"  True Positives  (yes -> yes): {cm[1,1]:,}")
print()
print("Feature: age")
print(f"  min={df['age'].min()}  max={df['age'].max()}")
print()
print("Categorical features and categories (excluding unknown):")
for feat in CATEGORICAL_FEATURES:
    cats = sorted(df[feat].dropna().unique().tolist())
    cats = [c for c in cats if c != "unknown"]
    print(f"  {feat:<15}: {cats}")
print()
print("poutcome subscription rates:")
for cat, row in grp.iterrows():
    print(f"  {str(cat):<15}: {row['rate']}%")
print("===========================")
