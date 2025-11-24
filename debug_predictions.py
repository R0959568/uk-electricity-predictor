"""
Debug script to test if the model is making real predictions or returning memorized values
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import sys

print("="*80)
print("DEBUGGING PREDICTION ISSUES - Testing if model is actually predicting")
print("="*80)

# Load model
model_path = Path('data/final/models/gradient_boosting_enhanced.pkl')
if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

model = joblib.load(model_path)
print(f"✅ Model loaded: {type(model).__name__}")

# Load historical data
hist_path = Path('data/interim/elec_cleaned_full.parquet')
if hist_path.exists():
    hist_df = pd.read_parquet(hist_path)
    hist_df['datetime'] = pd.to_datetime(hist_df['settlement_date'])
    print(f"✅ Historical data loaded: {len(hist_df):,} records")
    print(f"   Date range: {hist_df['datetime'].min()} to {hist_df['datetime'].max()}")
else:
    print("❌ Historical data not found")
    hist_df = None

# Function to create features (simplified from app)
def create_features(dt, hist_df):
    """Create features for a given datetime"""
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    day_of_week = dt.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = dt.isocalendar()[1]
    
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    is_peak_morning = 1 if (7 <= hour <= 9) else 0
    is_peak_evening = 1 if (17 <= hour <= 20) else 0
    
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season = season_map[month]
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Holidays (simplified)
    is_holiday = 0
    is_day_before_holiday = 0
    is_day_after_holiday = 0
    
    # Interactions
    weekend_hour = is_weekend * hour
    holiday_hour = is_holiday * hour
    month_hour = month * hour
    
    # Get lag features from historical data
    typical_demand = 35000
    if is_peak_morning or is_peak_evening:
        typical_demand = 42000
    elif is_night:
        typical_demand = 28000
    if season == 0:
        typical_demand *= 1.1
    elif season == 2:
        typical_demand *= 0.9
    if is_weekend:
        typical_demand *= 0.95
    
    demand_lag_1 = typical_demand
    demand_lag_1d = typical_demand
    demand_lag_3h = typical_demand
    demand_lag_7d = typical_demand
    demand_rolling_mean_24h = typical_demand
    demand_rolling_std_24h = 2500
    demand_rolling_mean_7d = typical_demand
    demand_diff_from_24h_avg = 0
    
    if hist_df is not None:
        hist_before = hist_df[hist_df['datetime'] < dt]
        if len(hist_before) > 0:
            demand_lag_1 = hist_before.iloc[-1]['demand_value']
            
            lag_1d_time = dt - timedelta(hours=24)
            lag_1d_record = hist_before[hist_before['datetime'] <= lag_1d_time]
            if len(lag_1d_record) > 0:
                demand_lag_1d = lag_1d_record.iloc[-1]['demand_value']
            
            rolling_24h_time = dt - timedelta(hours=24)
            rolling_24h_data = hist_before[hist_before['datetime'] >= rolling_24h_time]
            if len(rolling_24h_data) > 0:
                demand_rolling_mean_24h = rolling_24h_data['demand_value'].mean()
                demand_rolling_std_24h = rolling_24h_data['demand_value'].std()
                if pd.isna(demand_rolling_std_24h):
                    demand_rolling_std_24h = 0
                demand_diff_from_24h_avg = demand_lag_1 - demand_rolling_mean_24h
    
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

# Test scenarios
print("\n" + "="*80)
print("TESTING DIFFERENT SCENARIOS")
print("="*80)

test_cases = [
    ("2024-01-15 08:00", "Monday morning, winter, peak"),
    ("2024-01-15 14:00", "Monday afternoon, winter"),
    ("2024-01-15 23:00", "Monday night, winter"),
    ("2024-07-15 08:00", "Monday morning, summer, peak"),
    ("2024-07-15 14:00", "Monday afternoon, summer"),
    ("2024-07-15 23:00", "Monday night, summer"),
    ("2024-01-20 08:00", "Saturday morning, winter"),
    ("2024-01-20 14:00", "Saturday afternoon, winter"),
    ("2026-01-15 08:00", "Future: Monday morning, winter"),
    ("2026-07-15 14:00", "Future: Monday afternoon, summer"),
]

predictions = []
for dt_str, description in test_cases:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    features = create_features(dt, hist_df)
    pred = model.predict(features)[0]
    predictions.append(pred)
    
    print(f"\n{dt_str} ({description})")
    print(f"  Prediction: {pred:,.0f} MW")
    print(f"  Lag features:")
    print(f"    - demand_lag_1: {features['demand_lag_1'].values[0]:,.0f} MW")
    print(f"    - demand_lag_1d: {features['demand_lag_1d'].values[0]:,.0f} MW")
    print(f"    - rolling_24h_mean: {features['demand_rolling_mean_24h'].values[0]:,.0f} MW")

# Analysis
print("\n" + "="*80)
print("PREDICTION ANALYSIS")
print("="*80)
print(f"Minimum prediction: {min(predictions):,.0f} MW")
print(f"Maximum prediction: {max(predictions):,.0f} MW")
print(f"Range: {max(predictions) - min(predictions):,.0f} MW")
print(f"Mean: {np.mean(predictions):,.0f} MW")
print(f"Std Dev: {np.std(predictions):,.0f} MW")
print(f"Unique values: {len(set([int(p) for p in predictions]))}")

if max(predictions) - min(predictions) < 100:
    print("\n❌ PROBLEM DETECTED: Predictions have very low variance (<100 MW)")
    print("   This suggests the model is returning memorized values!")
    print("\n   Possible causes:")
    print("   1. Lag features dominating and all set to same typical value")
    print("   2. Model overfitting to lag features")
    print("   3. Feature scaling/normalization issues")
elif max(predictions) - min(predictions) < 5000:
    print("\n⚠️  WARNING: Predictions have low variance (<5,000 MW)")
    print("   This is lower than expected for UK electricity demand")
else:
    print("\n✅ Predictions show reasonable variance")
    print(f"   Range of {max(predictions) - min(predictions):,.0f} MW is realistic for UK demand")

# Check if predictions are close to 19968
if any(abs(p - 19968) < 1000 for p in predictions):
    print(f"\n🔍 FOUND: Some predictions are close to 19,968 MW")
    close_predictions = [p for p in predictions if abs(p - 19968) < 1000]
    print(f"   {len(close_predictions)} out of {len(predictions)} predictions near 19,968")
    print("   This value might be:")
    print("   - A default/fallback value in the code")
    print("   - The mean of training data")
    print("   - Result of specific lag feature values")
