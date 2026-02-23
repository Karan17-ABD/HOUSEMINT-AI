import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset (make sure the file is in your static folder)
file_path = "../static/Bengaluru_House_Data (1).csv"
df = pd.read_csv(file_path)

# ---------------- CLEANING ----------------

# Drop unnecessary columns if present
drop_cols = ["society", "availability"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Drop rows with missing essential values
df = df.dropna(subset=["location", "size", "bath", "total_sqft", "price"])

# Extract BHK from "size" column (e.g., "2 BHK" → 2)
df["BHK"] = df["size"].apply(lambda x: int(str(x).split(" ")[0]) if pd.notnull(x) else np.nan)

# Handle sqft ranges like "2100-2850"
def convert_sqft(x):
    try:
        if "-" in str(x):
            tokens = str(x).split("-")
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft_num"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft_num"])

# Price already in lakhs → keep it
df["price_lakhs"] = df["price"]

# Price per sqft (lakhs * 100000 / sqft)
df["price_per_sqft"] = (df["price_lakhs"] * 100000) / df["total_sqft_num"]

# Remove extreme outliers
df = df[(df["price_per_sqft"] > 2000) & (df["price_per_sqft"] < 15000)]

# Normalize location names (strip spaces, lowercase)
df["location"] = df["location"].apply(lambda x: x.strip().lower())

# Keep only locations with at least 10 data points
loc_counts = df["location"].value_counts()
df["location"] = df["location"].apply(lambda x: x if loc_counts[x] > 10 else "other")

# Drop unrealistic values (like too small sqft per BHK)
df = df[(df["total_sqft_num"] / df["BHK"]) >= 300]

# ---------------- SPLIT DATA ----------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------- SAVE ----------------
df.to_csv("../static/bengaluru_cleaned.csv", index=False)
train_df.to_csv("../static/bengaluru_train_cleaned.csv", index=False)
test_df.to_csv("../static/bengaluru_test_cleaned.csv", index=False)

print("✅ Clean dataset created successfully!")
print(" - Rows:", df.shape[0], "Columns:", df.shape[1])
print(" - Saved at: ../static/")
