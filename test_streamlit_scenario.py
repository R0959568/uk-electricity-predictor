"""
Test the exact datetime you're using in Streamlit to see if something specific causes 19968
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

print("="*80)
print("TESTING STREAMLIT PREDICTION SCENARIO")
print("="*80)

# Load model and data
model = joblib.load('data/final/models/gradient_boosting_enhanced.pkl')
hist_df = pd.read_parquet('data/interim/elec_cleaned_full.parquet')
hist_df['datetime'] = pd.to_datetime(hist_df['settlement_date'])

print(f"✅ Model and data loaded")
print(f"   Historical data: {len(hist_df):,} records")
print(f"   Latest data point: {hist_df['datetime'].max()}")

# Import the actual compute_enhanced_features function
import sys
sys.path.insert(0, '.')

# Recreate compute_enhanced_features from app
def compute_enhanced_features(prediction_datetime, historical_df):
    """Compute all enhanced features for a single prediction"""
    # Extract datetime components
    year = prediction_datetime.year
    month = prediction_datetime.month
    day = prediction_datetime.day
    hour = prediction_datetime.hour
    day_of_week = prediction_datetime.weekday()
    quarter = (month - 1) // 3 + 1
    week_of_year = prediction_datetime.isocalendar()[1]
    
    # Binary indicators
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    is_peak_morning = 1 if (7 <= hour <= 9) else 0
    is_peak_evening = 1 if (17 <= hour <= 20) else 0
    
    # Season mapping
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season = season_map[month]
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # UK Bank Holidays
    uk_holidays = [
        '2023-01-01', '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-29', 
        '2023-08-28', '2023-12-25', '2023-12-26',
        '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06', '2024-05-27', 
        '2024-08-26', '2024-12-25', '2024-12-26',
        '2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', 
        '2025-08-25', '2025-12-25', '2025-12-26'
    ]
    
    date_str = prediction_datetime.strftime('%Y-%m-%d')
    is_holiday = 1 if date_str in uk_holidays else 0
    
    day_before = (prediction_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
    day_after = (prediction_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
    is_day_before_holiday = 1 if day_before in uk_holidays else 0
    is_day_after_holiday = 1 if day_after in uk_holidays else 0
    
    weekend_hour = is_weekend * hour
    holiday_hour = is_holiday * hour
    month_hour = month * hour
    
    # Default typical demand values
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
    
    if historical_df is not None:
        hist_before = historical_df[historical_df['datetime'] < prediction_datetime]
        
        if len(hist_before) > 0:
            if len(hist_before) >= 1:
                demand_lag_1 = hist_before.iloc[-1]['demand_value']
            
            lag_1d_time = prediction_datetime - timedelta(hours=24)
            lag_1d_record = hist_before[hist_before['datetime'] <= lag_1d_time]
            if len(lag_1d_record) > 0:
                demand_lag_1d = lag_1d_record.iloc[-1]['demand_value']
            
            lag_3h_time = prediction_datetime - timedelta(hours=3)
            lag_3h_record = hist_before[hist_before['datetime'] <= lag_3h_time]
            if len(lag_3h_record) > 0:
                demand_lag_3h = lag_3h_record.iloc[-1]['demand_value']
            
            lag_7d_time = prediction_datetime - timedelta(days=7)
            lag_7d_record = hist_before[hist_before['datetime'] <= lag_7d_time]
            if len(lag_7d_record) > 0:
                demand_lag_7d = lag_7d_record.iloc[-1]['demand_value']
            
            rolling_24h_time = prediction_datetime - timedelta(hours=24)
            rolling_24h_data = hist_before[hist_before['datetime'] >= rolling_24h_time]
            if len(rolling_24h_data) > 0:
                demand_rolling_mean_24h = rolling_24h_data['demand_value'].mean()
                demand_rolling_std_24h = rolling_24h_data['demand_value'].std()
                if pd.isna(demand_rolling_std_24h):
                    demand_rolling_std_24h = 0
                demand_diff_from_24h_avg = demand_lag_1 - demand_rolling_mean_24h
            
            rolling_7d_time = prediction_datetime - timedelta(days=7)
            rolling_7d_data = hist_before[hist_before['datetime'] >= rolling_7d_time]
            if len(rolling_7d_data) > 0:
                demand_rolling_mean_7d = rolling_7d_data['demand_value'].mean()
    
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

# Test various dates including TODAY
test_dates = [
    datetime.now(),  # Current time
    datetime(2024, 11, 24, 10, 0),  # Today morning
    datetime(2024, 11, 24, 14, 0),  # Today afternoon
    datetime(2024, 11, 24, 20, 0),  # Today evening
    datetime(2025, 1, 15, 10, 0),   # Future date
    datetime(2025, 12, 31, 10, 0),  # End of year
]

print("\n" + "="*80)
print("TESTING PREDICTIONS")
print("="*80)

for dt in test_dates:
    features = compute_enhanced_features(dt, hist_df)
    prediction = model.predict(features)[0]
    
    print(f"\n📅 {dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Prediction: {prediction:,.2f} MW")
    print(f"   Lag features:")
    print(f"     - demand_lag_1: {features['demand_lag_1'].values[0]:,.0f} MW")
    print(f"     - demand_lag_1d: {features['demand_lag_1d'].values[0]:,.0f} MW")
    print(f"     - rolling_24h_mean: {features['demand_rolling_mean_24h'].values[0]:,.0f} MW")
    
    if abs(prediction - 19968) < 100:
        print(f"   ⚠️  FOUND THE ISSUE! Prediction is close to 19,968 MW")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If you're seeing 19,968 MW, it might be:")
print("1. A specific date/time combination")
print("2. Streamlit caching an old prediction")
print("3. Browser cache showing stale data")
print("\nTry clearing Streamlit cache by pressing 'C' in the app or rerunning it")
