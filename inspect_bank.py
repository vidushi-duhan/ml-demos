import pandas as pd
import numpy as np

df = pd.read_csv("data/bank.csv")

CANDIDATES = ["age", "job", "marital", "education", "housing", "loan", "poutcome"]
TARGET = "y"

SEP = "=" * 60

# ── 1. Shape and columns ─────────────────────────────────────────────────────
print(SEP)
print("1. SHAPE AND COLUMNS")
print(SEP)
print(f"Total rows   : {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns      : {list(df.columns)}")

# ── 2. Target distribution ───────────────────────────────────────────────────
print()
print(SEP)
print("2. TARGET DISTRIBUTION")
print(SEP)
vc = df[TARGET].value_counts()
total = len(df)
for val, count in vc.items():
    print(f"  '{val}': {count:5d}  ({count/total*100:.2f}%)")
print(f"  Imbalance ratio  no:yes = {vc.get('no', vc.iloc[0]):,} : {vc.get('yes', vc.iloc[1]):,}  "
      f"({vc.get('no', vc.iloc[0])/vc.get('yes', vc.iloc[1]):.1f}x)")

# ── 3. Duration column ───────────────────────────────────────────────────────
print()
print(SEP)
print("3. DURATION COLUMN (leakage risk)")
print(SEP)
if "duration" in df.columns:
    d = df["duration"]
    print(f"  Present: YES")
    print(f"  min={d.min()}  max={d.max()}  mean={d.mean():.1f}  median={d.median():.1f}")
    print(f"  NOTE: Must be dropped before training — known after call ends, not before.")
else:
    print("  Present: NO")

# ── 4. Candidate features ─────────────────────────────────────────────────────
print()
print(SEP)
print("4. CANDIDATE FEATURE RANGES / CATEGORIES")
print(SEP)
for feat in CANDIDATES:
    col = df[feat]
    if not pd.api.types.is_numeric_dtype(col):
        cats = sorted([x for x in col.unique().tolist() if x is not None and str(x) != 'nan'])
        print(f"\n  {feat} ({col.dtype}):")
        for c in cats:
            n = (col == c).sum()
            print(f"    '{c}': {n} ({n/total*100:.1f}%)")
    else:
        print(f"\n  {feat} (numeric): min={col.min()}  max={col.max()}  "
              f"mean={col.mean():.1f}  unique={col.nunique()}")

# ── 5. Missing / 'unknown' values ────────────────────────────────────────────
print()
print(SEP)
print("5. MISSING AND 'UNKNOWN' VALUES")
print(SEP)
print(f"  {'Column':<15} {'NaN':>6} {'unknown':>9}")
print(f"  {'-'*15} {'-'*6} {'-'*9}")
for col in df.columns:
    nan_count = df[col].isna().sum()
    unk_count = (df[col] == "unknown").sum() if not pd.api.types.is_numeric_dtype(df[col]) else 0
    if nan_count > 0 or unk_count > 0:
        print(f"  {col:<15} {nan_count:>6} {unk_count:>9}")
print()
print("  (columns with 0 NaN and 0 unknown omitted)")
print()
print("  'unknown' counts in candidate features:")
for feat in CANDIDATES:
    if df[feat].dtype == object:
        n = (df[feat] == "unknown").sum()
        print(f"    {feat:<15}: {n} ({n/total*100:.1f}%)")

# ── 6. Subscription rate by category ─────────────────────────────────────────
print()
print(SEP)
print("6. SUBSCRIPTION RATE BY CATEGORY (signal check)")
print(SEP)

df["_subscribed"] = (df[TARGET] == "yes").astype(int)

for feat in CANDIDATES:
    print(f"\n  -- {feat} --")
    grp = (
        df.groupby(feat)["_subscribed"]
        .agg(count="count", subscribed="sum")
        .assign(rate_pct=lambda x: (x["subscribed"] / x["count"] * 100).round(1))
        .sort_values("rate_pct", ascending=False)
    )
    for cat, row in grp.iterrows():
        bar = "#" * int(row["rate_pct"] / 2)
        print(f"    {str(cat):<20} {row['rate_pct']:5.1f}%  {bar}  (n={int(row['count'])})")

df.drop(columns=["_subscribed"], inplace=True)

# ── 7. Summary block ─────────────────────────────────────────────────────────
print()
print(SEP)
print("=== SUMMARY FOR FRONTEND ===")
print(f"Total rows: {len(df)}")

vc = df[TARGET].value_counts()
yes_n = int(vc.get("yes", 0))
no_n  = int(vc.get("no",  0))
print(f"Class balance: yes={yes_n} ({yes_n/total*100:.1f}%)  no={no_n} ({no_n/total*100:.1f}%)")

if "duration" in df.columns:
    d = df["duration"]
    print(f"Duration column: present  min={d.min()} max={d.max()} mean={d.mean():.1f}  --> DROP before training")
else:
    print("Duration column: not present")

print()
print("Missing/unknown values per candidate feature:")
for feat in CANDIDATES:
    if df[feat].dtype == object:
        n = (df[feat] == "unknown").sum()
        print(f"  {feat:<15}: {n} unknown ({n/total*100:.1f}%)")
    else:
        print(f"  {feat:<15}: 0 (numeric, no unknowns)")

print()
print("Candidate feature ranges:")
for feat in CANDIDATES:
    col = df[feat]
    if not pd.api.types.is_numeric_dtype(col):
        cats = sorted([c for c in col.unique().tolist() if c is not None and str(c) != 'nan' and c != "unknown"])
        print(f"  {feat:<15}: {cats}")
    else:
        print(f"  {feat:<15}: min={col.min()}  max={col.max()}")

print()
print("Subscription rate by category (quick signal check):")
df["_subscribed"] = (df[TARGET] == "yes").astype(int)
for feat in CANDIDATES:
    grp = (
        df.groupby(feat)["_subscribed"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    print(f"  {feat}: {dict(grp)}")
df.drop(columns=["_subscribed"], inplace=True)

print("===========================")
