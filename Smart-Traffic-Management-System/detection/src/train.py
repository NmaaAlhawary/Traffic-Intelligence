import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("mirpur10_traffic_sample.csv")

# Encode weather_condition
df["weather_condition"] = LabelEncoder().fit_transform(df["weather_condition"])

# Feature columns
features = [
    "hour", "dayofweek", "is_weekend",
    "temperature", "humidity", "rain", "wind_speed",
    "vehicle_count_last_5min", "weather_condition"
]
target = "vehicle_count"

# Prepare feature matrix and labels
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained. MSE: {mse:.2f}")

# Save model and scaler
joblib.dump(model, "traffic_predictor.pkl")
joblib.dump(scaler, "traffic_scaler.pkl")
print("✅ Model and scaler saved.")