import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

os.makedirs("data", exist_ok=True)

# Heart Disease (id=45)
heart = fetch_ucirepo(id=45)
df_heart = pd.concat([heart.data.features, heart.data.targets], axis=1)
df_heart.to_csv("data/heart.csv", index=False)
print(f"Saved data/heart.csv: {df_heart.shape[0]} rows, {df_heart.shape[1]} columns")

# Student Performance (id=320) — Portuguese class, 649 students
student = fetch_ucirepo(id=320)
# ucimlrepo returns both Math and Portuguese combined; filter to por file (649 rows)
df_student = pd.concat([student.data.features, student.data.targets], axis=1)
# The dataset has a 'school' column but both files are already separate in id=320
# por file is identifiable by 649 rows — take first 649 to be safe if combined
if len(df_student) > 649:
    df_student = df_student.iloc[:649].reset_index(drop=True)
df_student.to_csv("data/student.csv", index=False)
print(f"Saved data/student.csv: {df_student.shape[0]} rows, {df_student.shape[1]} columns")

# Bank Marketing (id=222) — full dataset (45,211 rows) to retain poutcome
bank = fetch_ucirepo(id=222)
df_bank = pd.concat([bank.data.features, bank.data.targets], axis=1)
df_bank.to_csv("data/bank.csv", index=False)
print(f"Saved data/bank.csv: {df_bank.shape[0]} rows, {df_bank.shape[1]} columns")
