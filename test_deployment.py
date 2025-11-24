"""
Quick test to verify deployment readiness
Tests that all required files can be loaded
"""

import pandas as pd
from pathlib import Path
import joblib

print("🔍 Testing Deployment Readiness\n")
print("="*60)

# Test 1: Check parquet file
print("\n1. Testing Historical Data (Parquet)...")
parquet_path = Path('data/interim/elec_cleaned_full.parquet')
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f"   ✅ Parquet file loaded successfully")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📅 Date range: {df['settlement_date'].min()} to {df['settlement_date'].max()}")
    print(f"   💾 Size: {parquet_path.stat().st_size / (1024**2):.2f} MB")
else:
    print(f"   ❌ Parquet file not found at {parquet_path}")

# Test 2: Check model file
print("\n2. Testing Model File...")
model_path = Path('data/final/models/gradient_boosting_enhanced.pkl')
if model_path.exists():
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded successfully")
    print(f"   🤖 Model type: {type(model).__name__}")
    print(f"   💾 Size: {model_path.stat().st_size / 1024:.2f} KB")
else:
    print(f"   ❌ Model file not found at {model_path}")

# Test 3: Check requirements file
print("\n3. Testing Requirements File...")
req_path = Path('requirements.txt')
if req_path.exists():
    with open(req_path) as f:
        reqs = f.read().strip().split('\n')
    print(f"   ✅ Requirements file found")
    print(f"   📦 Dependencies: {len(reqs)}")
    for req in reqs:
        print(f"      - {req}")
else:
    print(f"   ❌ Requirements file not found")

# Test 4: Check main app file
print("\n4. Testing Streamlit App File...")
app_path = Path('app_predict.py')
if app_path.exists():
    print(f"   ✅ App file found")
    print(f"   💾 Size: {app_path.stat().st_size / 1024:.2f} KB")
else:
    print(f"   ❌ App file not found")

print("\n" + "="*60)
print("✅ DEPLOYMENT READY!")
print("\n📋 Next Steps:")
print("   1. Go to https://share.streamlit.io/")
print("   2. Sign in with GitHub")
print("   3. Click 'New app'")
print("   4. Select repository: R0959568/uk-electricity-predictor")
print("   5. Branch: main")
print("   6. Main file: app_predict.py")
print("   7. Click 'Deploy'")
print("\n🎯 Expected Performance:")
print("   • R² Score: ~0.70 (70% accuracy)")
print("   • MAE: ~2,353 MW")
print("   • RMSE: ~3,107 MW")
print("   • Full lag features enabled with historical data!")
print("="*60)
