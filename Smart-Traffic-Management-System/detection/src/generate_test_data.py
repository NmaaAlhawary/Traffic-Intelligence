import pandas as pd
import random
from datetime import datetime, timedelta

# Simulate 2 days of data at 5-minute intervals
rows = []
start = datetime(2025, 7, 15, 6, 0)
for i in range(10 * 24 * 12):  # 10 days, every 5 mins
    timestamp = start + timedelta(minutes=5 * i)
    hour = timestamp.hour
    dayofweek = timestamp.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    temperature = round(random.uniform(28, 34), 1)
    humidity = round(random.uniform(55, 90), 1)
    rain = round(random.choice([0.0, 0.1, 0.3, 0.5]), 1)
    wind_speed = round(random.uniform(5, 15), 1)
    weather_condition = random.choices(
        ["Clear", "Cloudy", "Rain"], weights=[0.5, 0.3, 0.2]
    )[0]
    vc_last = random.randint(50, 150)
    vc = vc_last + random.randint(-5, 5)

    rows.append([
        timestamp, "Mirpur 10", hour, dayofweek, is_weekend, temperature,
        humidity, rain, wind_speed, weather_condition, vc_last, max(vc, 0)
    ])

df = pd.DataFrame(rows, columns=[
    "timestamp", "location", "hour", "dayofweek", "is_weekend",
    "temperature", "humidity", "rain", "wind_speed", "weather_condition",
    "vehicle_count_last_5min", "vehicle_count"
])

df.to_csv("mirpur10_traffic_sample.csv", index=False)
print("✅ Sample dataset saved as 'mirpur10_traffic_sample.csv'")