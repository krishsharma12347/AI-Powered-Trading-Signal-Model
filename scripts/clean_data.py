import pandas as pd
import os

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

CLEANED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "cleaned_btc_5m.csv"
)

# =========================
# GET ALL CSV FILES
# =========================

csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")]

print("\n5m CSV Files Found:")
print(csv_files)

# =========================
# LOAD & CLEAN DATA
# =========================

all_dataframes = []

for file in csv_files:

    file_path = os.path.join(RAW_DATA_PATH, file)

    print(f"\nLoading File: {file}")

    df = pd.read_csv(file_path)

    print(f"Original Shape: {df.shape}")

    # =========================
    # REMOVE FULLY EMPTY COLUMNS
    # =========================

    df.dropna(axis=1, how='all', inplace=True)

    # =========================
    # FORWARD FILL
    # =========================

    df.ffill(inplace=True)

    # =========================
    # DROP REMAINING NaNs
    # =========================

    df.dropna(inplace=True)

    # =========================
    # REMOVE DUPLICATES
    # =========================

    df.drop_duplicates(inplace=True)

    print(f"Cleaned Shape: {df.shape}")

    # ADD CLEANED DATAFRAME
    all_dataframes.append(df)


# =========================
# KEEP COMMON COLUMNS ONLY
# =========================

common_columns = set(all_dataframes[0].columns)

for df in all_dataframes[1:]:
    common_columns = common_columns.intersection(df.columns)

common_columns = list(common_columns)

print("\nCommon Columns:")
print(common_columns)

# KEEP ONLY COMMON COLUMNS

all_dataframes = [df[common_columns] for df in all_dataframes]

# =========================
# COMBINE ALL YEARS
# =========================

master_df = pd.concat(all_dataframes, ignore_index=True)

# =========================
# FINAL CLEANING
# =========================

master_df.drop_duplicates(inplace=True)

master_df.reset_index(drop=True, inplace=True)

# =========================
# FINAL INFO
# =========================

print("\n=========================")
print("FINAL CLEANED DATASET")
print("=========================")

print(f"Final Shape: {master_df.shape}")

print(f"Remaining NaNs: {master_df.isna().sum().sum()}")

# =========================
# SAVE CLEANED CSV
# =========================

master_df.to_csv(CLEANED_FILE, index=False)

print("\n=========================")
print("CLEANING COMPLETED")
print("=========================")

print(f"\nCleaned dataset saved at:")
print(CLEANED_FILE)