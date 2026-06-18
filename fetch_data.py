import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=45)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart.csv", index=False)

print(f"Saved data/heart.csv: {df.shape[0]} rows, {df.shape[1]} columns")
