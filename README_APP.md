# UK Electricity Demand Prediction App

Interactive web application for predicting UK electricity demand using trained machine learning models.

## Features

- **Single Prediction**: Predict demand for any specific date and time
- **24-Hour Forecast**: Generate predictions for the next 24 hours
- **Model Comparison**: Compare performance of different ML models
- **Interactive Visualizations**: Gauge charts, line plots, and more
- **Model Selection**: Choose between Linear Regression, Random Forest, or Gradient Boosting

## Installation

1. Install required packages:
```bash
pip install -r requirements_app.txt
```

2. Ensure trained models are available in `../data/final/models/`

## Running the App

From the `dataset_2_electricity` directory, run:

```bash
streamlit run app_predict.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

### Single Prediction Tab
1. Select a date and time
2. View derived features (day of week, quarter, etc.)
3. Click "Predict Demand" to get the forecast
4. See prediction with context (low/normal/high/peak demand)

### Batch Prediction Tab
1. Select start date and hour
2. Click "Generate 24-Hour Forecast"
3. View statistics and interactive chart
4. Download forecast as CSV

### Model Comparison Tab
- View metrics for all trained models
- Compare MAE, RMSE, and R² scores
- Identify the best performing model

## Model Performance

- **Gradient Boosting**: R² = 0.6091 (Best)
- **Random Forest**: R² = 0.6068
- **Linear Regression**: R² = 0.5532

## Data

- Historical UK electricity demand data (2001-2025)
- Features: Year, Month, Day, Hour, Day of Week, and cyclical encodings
- Target: Electricity demand in Megawatts (MW)

## Technologies

- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data processing
