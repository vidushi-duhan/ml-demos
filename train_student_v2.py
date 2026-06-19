import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error

FEATURES = ["studytime", "failures", "absences", "Medu", "Fedu", "goout", "higher"]
TARGET = "G3"

BASELINES = {
    "Linear Regression": 0.2004,
    "Ridge":             0.2005,
    "Lasso":             0.0312,
}

def encode(df):
    d = df[FEATURES + [TARGET]].copy()
    d["higher"] = (d["higher"] == "yes").astype(int)
    return d

def run_models(df_model, label):
    X = df_model[FEATURES]
    y = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    specs = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=1.0),
    }
    results = {}
    print(f"\n--- {label} ---")
    for name, reg in specs.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        results[name] = r2
        print(f"  {name:20s}  R2={r2:.4f}  RMSE={rmse:.4f}  CV_R2={cv.mean():.4f}+-{cv.std():.4f}")
    return results

# ── Load ─────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("data/student.csv")
df_full = encode(df_raw)

# ── Filtered ─────────────────────────────────────────────────────────────────
df_filt = df_full[df_full[TARGET] != 0].reset_index(drop=True)
removed = len(df_full) - len(df_filt)
print(f"Rows with G3=0 removed : {removed}")
print(f"Rows remaining         : {len(df_filt)}")

# ── Train both ───────────────────────────────────────────────────────────────
r2_with    = run_models(df_full, "WITH zeros (G3=0 included, n=649)")
r2_without = run_models(df_filt, "WITHOUT zeros (G3=0 removed, n=634)")

# ── Before / after table ─────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Model':<22} {'R2 with zeros':>14} {'R2 without zeros':>17} {'Delta':>7}")
print("-" * 60)
for name in r2_with:
    before = r2_with[name]
    after  = r2_without[name]
    delta  = after - before
    sign   = "+" if delta >= 0 else ""
    print(f"  {name:<20} {before:>14.4f} {after:>17.4f} {sign}{delta:>6.4f}")
print("=" * 60)

# ── Verdict ──────────────────────────────────────────────────────────────────
print()
best_delta = max(r2_without[n] - r2_with[n] for n in r2_with)
if best_delta > 0.03:
    print("Verdict: Removing dropouts IMPROVED the model meaningfully (>0.03 R2 gain).")
    print("         Consider training the final model on the filtered dataset.")
elif best_delta > 0:
    print("Verdict: Removing dropouts gave a marginal improvement (<0.03 R2 gain).")
    print("         Not enough to justify filtering — keep all 649 rows.")
else:
    print("Verdict: Removing dropouts did NOT improve the model.")
    print("         Keep all 649 rows for the final model.")
