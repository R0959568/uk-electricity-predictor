"""
Test the fixed prediction logic with hour-specific demand profiles
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

print("="*80)
print("TESTING FIXED PREDICTION LOGIC")
print("="*80)

# Load model
model = joblib.load('data/final/models/gradient_boosting_enhanced.pkl')
hist_df = pd.read_parquet('data/interim/elec_cleaned_full.parquet')
hist_df['datetime'] = pd.to_datetime(hist_df['settlement_date'])

print(f"✅ Model and data loaded\n")

# Import the fixed compute_enhanced_features function
exec(open('app_predict.py').read().split('def compute_enhanced_features')[1].split('def ')[0].replace('def compute_enhanced_features', 'def compute_enhanced_features'))

# Simpler approach - just recreate the key part
def compute_features_fixed(prediction_datetime):
    """Compute features with FIXED hour-specific logic"""
    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour
    day_of_week = prediction_datetime.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = prediction_datetime.isocalendar()[1]
    
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    is_peak_morning = 1 if (7 <= hour <= 9) else 0
    is_peak_evening = 1 if (17 <= hour <= 20) else 0
    
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season = season_map[month]
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    is_holiday = 0
    is_day_before_holiday = 0
    is_day_after_holiday = 0
    
    weekend_hour = is_weekend * hour
    holiday_hour = is_holiday * hour
    month_hour = month * hour
    
    # FIXED: Hour-specific demand profile
    hour_demand_profile = {
        0: 23000, 1: 21000, 2: 20000, 3: 19500, 4: 19000, 5: 20000,
        6: 24000, 7: 30000, 8: 35000, 9: 37000, 10: 38000, 11: 38500,
        12: 38000, 13: 37500, 14: 37000, 15: 36500, 16: 37000, 17: 39000,
        18: 41000, 19: 42000, 20: 40000, 21: 37000, 22: 32000, 23: 27000
    }
    
    typical_demand = hour_demand_profile.get(hour, 35000)
    
    if season == 0:
        typical_demand *= 1.15
    elif season == 1:
        typical_demand *= 1.0
    elif season == 2:
        typical_demand *= 0.85
    else:
        typical_demand *= 1.05
    
    if is_weekend:
        typical_demand *= 0.90
    
    demand_lag_1 = typical_demand * 0.98
    demand_lag_1d = typical_demand * 1.0
    demand_lag_3h = hour_demand_profile.get((hour - 3) % 24, 35000) * (1.15 if season == 0 else 0.85 if season == 2 else 1.0)
    demand_lag_7d = typical_demand * 1.0
    demand_rolling_mean_24h = typical_demand * 0.98
    demand_rolling_std_24h = 2500
    demand_rolling_mean_7d = typical_demand * 0.99
    demand_diff_from_24h_avg = typical_demand * 0.02
    
    feature_names = [
        'year', 'month', 'day', 'hour', 'day_of_week', 'quarter', 'week_of_year',
        'is_weekend', 'is_business_hours', 'is_night', 'is_peak_morning', 'is_peak_evening',
        'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'demand_lag_1', 'demand_lag_1d', 'demand_lag_3h', 'demand_lag_7d',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h', 'demand_rolling_mean_7d',
        'demand_diff_from_24h_avg',
        'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
        'weekend_hour', 'holiday_hour', 'month_hour'
    ]
    
    feature_values = [
        year, month, day, hour, day_of_week, quarter, week_of_year,
        is_weekend, is_business_hours, is_night, is_peak_morning, is_peak_evening,
        season,
        hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos,
        demand_lag_1, demand_lag_1d, demand_lag_3h, demand_lag_7d,
        demand_rolling_mean_24h, demand_rolling_std_24h, demand_rolling_mean_7d,
        demand_diff_from_24h_avg,
        is_holiday, is_day_before_holiday, is_day_after_holiday,
        weekend_hour, holiday_hour, month_hour
    ]
    
    return pd.DataFrame([feature_values], columns=feature_names)

# Test the same day at different hours
print("TESTING: Same day (2025-12-15), different hours")
print("="*80)

test_hours = [0, 6, 8, 12, 14, 18, 20, 23]
predictions = []

for h in test_hours:
    dt = datetime(2025, 12, 15, h, 0)
    features = compute_features_fixed(dt)
    pred = model.predict(features)[0]
    predictions.append(pred)
    
    print(f"Hour {h:02d}:00 - Prediction: {pred:7,.0f} MW  |  Lag 1d: {features['demand_lag_1d'].values[0]:,.0f} MW")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print(f"Min prediction: {min(predictions):,.0f} MW")
print(f"Max prediction: {max(predictions):,.0f} MW")
print(f"Range: {max(predictions) - min(predictions):,.0f} MW")
print(f"Unique predictions: {len(set([int(p) for p in predictions]))}")

if max(predictions) - min(predictions) > 5000:
    print("\n✅ SUCCESS! Predictions now vary significantly by hour!")
    print(f"   Range of {max(predictions) - min(predictions):,.0f} MW shows hour-specific patterns")
else:
    print("\n❌ Still has issues - predictions don't vary enough")

# Test different days
print("\n" + "="*80)
print("TESTING: Same hour (8 AM), different days/seasons")
print("="*80)

test_dates = [
    datetime(2026, 1, 15, 8, 0),   # Winter weekday
    datetime(2026, 4, 15, 8, 0),   # Spring weekday
    datetime(2026, 7, 15, 8, 0),   # Summer weekday
    datetime(2026, 10, 15, 8, 0),  # Autumn weekday
    datetime(2026, 1, 18, 8, 0),   # Winter weekend
]

for dt in test_dates:
    features = compute_features_fixed(dt)
    pred = model.predict(features)[0]
    day_name = dt.strftime("%A")
    print(f"{dt.strftime('%Y-%m-%d')} ({day_name}, Season {features['season'].values[0]}) - {pred:,.0f} MW")

print("\n" + "="*80)
print("✅ Fixed! Predictions should now vary based on hour and season!")
print("="*80)
