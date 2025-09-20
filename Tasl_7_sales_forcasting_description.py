import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor

# ---------------- Dataset Path ----------------
base_path = r"C:\Users\NCS\Documents\Machine_Learning\walmart_dataset"

train = pd.read_csv(os.path.join(base_path, "train.csv"), parse_dates=["Date"])
test = pd.read_csv(os.path.join(base_path, "test.csv"), parse_dates=["Date"])
features = pd.read_csv(os.path.join(base_path, "features.csv"), parse_dates=["Date"])
stores = pd.read_csv(os.path.join(base_path, "stores.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ---------------- Merge Files ----------------
train = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
train = train.merge(stores, on="Store", how="left")

test = test.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
test = test.merge(stores, on="Store", how="left")

print("Train merged shape:", train.shape)
print("Test merged shape:", test.shape)

# ---------------- Feature Engineering ----------------
for df in [train, test]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

# ---------------- Handle Missing Values ----------------
markdown_cols = [col for col in train.columns if "MarkDown" in col]
for df in [train, test]:
    df[markdown_cols] = df[markdown_cols].fillna(0)

# ---------------- Encode Categorical ----------------
le = LabelEncoder()
for df in [train, test]:
    if 'Type' in df.columns:
        df['Type'] = le.fit_transform(df['Type'])

# ---------------- Split Input/Target ----------------
X = train.drop(columns=['Weekly_Sales', 'Date'])
y = train['Weekly_Sales']
X_test_final = test.drop(columns=['Date'])

# ---------------- Normalization ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# ---------------- Train / Validation Split ----------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------- Model Setup ----------------
if LGB_AVAILABLE:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
else:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

# ---------------- Train Model ----------------
model.fit(X_train, y_train)

# ---------------- Validation Prediction ----------------
val_preds = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae = mean_absolute_error(y_val, val_preds)
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE: {mae:.2f}")

# ---------------- Train Final Model ----------------
model.fit(X_scaled, y)
train_preds = model.predict(X_scaled)

# ---------------- Predict on Test ----------------
test_preds = model.predict(X_test_scaled)

# ---------------- Plot Sample Results ----------------
plt.figure(figsize=(12,6))
plt.plot(y.values[:200], label="Actual")
plt.plot(train_preds[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Sales (Sample 200 points)")
plt.show()
