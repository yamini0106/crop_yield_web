import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

print("Training the model in the same environment...")

# 1. Load Data
df = pd.read_csv('crop_yield_dataset.csv')

# 2. Preprocess
X = df.drop('Yield_ton_per_ha', axis=1)
y = df['Yield_ton_per_ha']

categorical_features = ['Crop', 'Region', 'Soil_Type', 'Irrigation', 'Previous_Crop']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# 3. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# 4. Save Model and Columns in the current environment
os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/crop_yield_model.pkl')
joblib.dump(X_encoded.columns, 'model/model_columns.pkl')

print("✅ SUCCESS: Model and columns saved in the 'model' folder!")