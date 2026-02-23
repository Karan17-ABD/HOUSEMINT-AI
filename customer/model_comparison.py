import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ================= PATHS =================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "static"))
DATA_PATH  = os.path.join(STATIC_DIR, "Bengaluru_House_Data_Enriched D.csv")

# ================= HELPERS =================
def convert_sqft(x):
    """Convert ranges like 1200-1500 to average"""
    try:
        tokens = str(x).split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

def clean_data():
    print(f"✅ Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw rows: {df.shape[0]} columns: {df.shape[1]}")

    # Drop duplicates
    df = df.drop_duplicates()

    # Only required cols
    keep_cols = ['location','size','total_sqft','bath','price']
    df = df[keep_cols]

    # Drop nulls
    df = df.dropna()

    # BHK feature
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if isinstance(x,str) else None)

    # Convert sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft'])

    # Remove unrealistic sqft per bhk
    df = df[~(df.total_sqft/df.bhk < 300)]

    # Price per sqft
    df['price_per_sqft'] = df['price']*100000/df['total_sqft']

    # Bathroom sanity check
    df = df[df.bath < df.bhk+3]

    # Engineered features
    df["bath_per_bhk"] = df["bath"] / df["bhk"]
    df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

    # Drop NaNs again after feature engg
    df = df.dropna()

    print(f"✅ Cleaned dataset shape: {df.shape}")
    cleaned_path = os.path.join(STATIC_DIR, "Bengaluru_House_Data_Enriched_D_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned to: {cleaned_path}")

    return df

def _prepare_features(df):
    X = df[["location","total_sqft","bhk","bath","bath_per_bhk","sqft_per_bhk"]]
    y = df["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save train/test
    pd.concat([X_train,y_train],axis=1).to_csv(os.path.join(STATIC_DIR,"bengaluru_train.csv"),index=False)
    pd.concat([X_test,y_test],axis=1).to_csv(os.path.join(STATIC_DIR,"bengaluru_test.csv"),index=False)

    # Preprocessor
    numeric = ["total_sqft","bhk","bath","bath_per_bhk","sqft_per_bhk"]
    categorical = ["location"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop")

    return X_train, X_test, y_train, y_test, preprocessor

def _models_dict():
    models = {
        "Linear Regression": (LinearRegression(), False),
        "Decision Tree": (DecisionTreeRegressor(random_state=42), True),
        "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), True),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, tree_method="hist"
        ), True)
    except ImportError:
        print("⚠️ XGBoost not installed. Run: pip install xgboost")
    return models

# ================= MAIN =================
def main():
    df = clean_data()
    X_train, X_test, y_train, y_test, preprocessor = _prepare_features(df)
    models = _models_dict()
    results = []

    print("\n📊 Model Performance (Accuracy in %):\n")

    for name, (reg, use_log) in models.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
        model = pipe

        model.fit(X_train, y_train)

        y_pred_tr = model.predict(X_train)
        y_pred_te = model.predict(X_test)

        r2_tr = r2_score(y_train, y_pred_tr) * 100
        r2_te = r2_score(y_test,  y_pred_te) * 100

        results.append([name, round(r2_tr,2), round(r2_te,2)])
        print(f"{name:16s} | Train Accuracy: {r2_tr:6.2f}% | Test Accuracy: {r2_te:6.2f}%")

    res_df = pd.DataFrame(results, columns=["Model","Train_Accuracy_%","Test_Accuracy_%"])
    res_path = os.path.join(STATIC_DIR,"model_performance.csv")
    res_df.to_csv(res_path, index=False)

    print("\n" + "="*35)
    print(res_df.to_string(index=False))
    print("="*35)

    # project main model = XGBoost
    xgb_row = res_df[res_df["Model"]=="XGBoost"].iloc[0]
    
    print(f"✅ Main Model: XGBoost with Train Accuracy {xgb_row['Train_Accuracy_%']}% "
          f"and Test Accuracy {xgb_row['Test_Accuracy_%']}%")

if __name__ == "__main__":
    main()
