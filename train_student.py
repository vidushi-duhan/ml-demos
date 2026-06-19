import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error

FEATURES = ["studytime", "failures", "absences", "Medu", "Fedu", "goout", "higher"]
TARGET = "G3"

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv("data/student.csv")

print("=" * 60)
print("INSPECTION REPORT")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"All columns ({len(df.columns)}): {list(df.columns)}")
print()

print("Candidate feature ranges and unique values:")
candidate_feats = ["studytime", "failures", "absences", "Medu", "Fedu", "goout", "higher"]
for feat in candidate_feats:
    col = df[feat]
    if col.dtype == object:
        print(f"  {feat:12s}  unique={col.nunique()}  values={sorted(col.unique().tolist())}")
    else:
        print(f"  {feat:12s}  min={col.min()}  max={col.max()}  unique={col.nunique()}")
print()

print(f"Target G3: min={df[TARGET].min()}  max={df[TARGET].max()}  mean={df[TARGET].mean():.2f}")
print()
print("G3 distribution (text histogram):")
bins = range(0, 22)
counts, edges = np.histogram(df[TARGET], bins=bins)
for i, c in enumerate(counts):
    bar = "#" * (c // 3)
    print(f"  {i:2d}: {bar} ({c})")
print()

print("G1 and G2 exist in dataset — EXCLUDED from training:")
print(f"  G1: min={df['G1'].min()}  max={df['G1'].max()}  (excluded)")
print(f"  G2: min={df['G2'].min()}  max={df['G2'].max()}  (excluded)")
print()

# ── 2. Prepare data ──────────────────────────────────────────────────────────
df_model = df[FEATURES + [TARGET]].copy()

print("Encoding 'higher' (yes/no -> 1/0):")
print(f"  Before: {dict(df_model['higher'].value_counts())}")
df_model["higher"] = (df_model["higher"] == "yes").astype(int)
print(f"  After:  {dict(df_model['higher'].value_counts())}  (yes=1, no=0)")
print()

print("Missing values per feature:")
for feat in FEATURES + [TARGET]:
    n = df_model[feat].isna().sum()
    print(f"  {feat:12s}: {n}")
print()

# ── 3. Train three models ────────────────────────────────────────────────────
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
    "Ridge":             Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
    "Lasso":             Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=1.0))]),
}

results = {}
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    cv = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    results[name] = {"r2": r2, "rmse": rmse, "cv_mean": cv.mean(), "cv_std": cv.std(), "pipe": pipe}
    print(f"{name}:")
    print(f"  Test R²  : {r2:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  CV R²    : {cv.mean():.4f} ± {cv.std():.4f}")
    print()

# ── 4. Best model ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["r2"])
print(f"Best model on test R²: {best_name}")
os.makedirs("models", exist_ok=True)
joblib.dump(results[best_name]["pipe"], "models/student_model.pkl")
print(f"Saved models/student_model.pkl  ({best_name})")
print()

# ── 5. Coefficients ──────────────────────────────────────────────────────────
print("=" * 60)
print("COEFFICIENTS")
print("=" * 60)
for name in ["Ridge", "Lasso"]:
    coefs = results[name]["pipe"].named_steps["reg"].coef_
    print(f"\n{name} coefficients:")
    for feat, coef in zip(FEATURES, coefs):
        zeroed = "  <-- ZEROED OUT" if abs(coef) < 1e-6 else ""
        print(f"  {feat:12s}: {coef:+.4f}{zeroed}")

# ── 6. Summary block ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("=== SUMMARY FOR FRONTEND ===")
print(f"Total students: {len(df_model)}")
print(f"Target G3 range: min={df[TARGET].min()}  max={df[TARGET].max()}  mean={df[TARGET].mean():.2f}")
total_missing = df_model[FEATURES].isna().sum().sum()
print(f"Missing values: {total_missing}")
print()
for name, r in results.items():
    print(f"{name}: R2={r['r2']:.4f}  RMSE={r['rmse']:.4f}  CV_R2={r['cv_mean']:.4f}±{r['cv_std']:.4f}")
print(f"Best model saved: {best_name}  ->  models/student_model.pkl")
print()
print("Feature ranges (for sliders):")
print(f"  studytime: min={df['studytime'].min()}  max={df['studytime'].max()}  (1=<2h, 2=2-5h, 3=5-10h, 4=>10h per week)")
print(f"  failures:  min={df['failures'].min()}  max={df['failures'].max()}  (number of past class failures)")
print(f"  absences:  min={df['absences'].min()}  max={df['absences'].max()}  (number of school absences)")
print(f"  Medu:      min={df['Medu'].min()}  max={df['Medu'].max()}  (0=none, 1=primary, 2=middle, 3=secondary, 4=higher)")
print(f"  Fedu:      min={df['Fedu'].min()}  max={df['Fedu'].max()}  (same scale as Medu)")
print(f"  goout:     min={df['goout'].min()}  max={df['goout'].max()}  (1=very low, 5=very high going out with friends)")
print(f"  higher:    yes=1, no=0  (wants to pursue higher education)")
print()
print("Ridge coefficients per feature:")
ridge_coefs = results["Ridge"]["pipe"].named_steps["reg"].coef_
for feat, coef in zip(FEATURES, ridge_coefs):
    print(f"  {feat:12s}: {coef:+.4f}")
print()
print("Lasso coefficients per feature (note any zeroed out):")
lasso_coefs = results["Lasso"]["pipe"].named_steps["reg"].coef_
for feat, coef in zip(FEATURES, lasso_coefs):
    zeroed = "  <-- zeroed" if abs(coef) < 1e-6 else ""
    print(f"  {feat:12s}: {coef:+.4f}{zeroed}")
print("===========================")
