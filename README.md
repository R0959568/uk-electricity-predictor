# UK Electricity Demand Predictor ⚡

Interactive Streamlit application for predicting UK electricity demand using machine learning.

## 🎯 Features

- **Real-time Predictions**: Predict electricity demand for any date and time
- **Advanced ML Model**: Gradient Boosting with 39 enhanced features
- **High Accuracy**: R² Score: 0.70, MAE: 2,353 MW, RMSE: 3,107 MW
- **Interactive UI**: Built with Streamlit for easy interaction
- **Historical Context**: Includes 13MB parquet file with 25 years of UK electricity data (2001-2025)
- **Full Lag Features**: Real historical demand data enables accurate time-series predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/R0959568/uk-electricity-predictor.git
cd uk-electricity-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app_predict.py
```

4. Open your browser at `http://localhost:8501`

## 📊 Model Performance

- **Model**: Gradient Boosting Regressor
- **Features**: 39 enhanced features including:
  - Temporal features (hour, day, month, year, season)
  - Lag features (historical demand patterns)
  - Rolling statistics (24h and 7d averages)
  - UK bank holidays
  - Cyclical encoding (sine/cosine transformations)
  - Interaction features

### Performance Metrics
- **R² Score**: 0.6999
- **MAE**: 2,353.23 MW
- **RMSE**: 3,107.24 MW

## 🛠️ Technology Stack

- **Python 3.12**
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **joblib**: Model serialization

## 📁 Project Structure

```
uk-electricity-predictor/
├── app_predict.py              # Main Streamlit application
├── data/
│   ├── final/
│   │   └── models/
│   │       └── gradient_boosting_enhanced.pkl  # Trained model (832KB)
│   └── interim/
│       └── elec_cleaned_full.parquet  # Historical data (13MB) ✅ INCLUDED
├── 02_models_local.ipynb       # Model training notebook
├── requirements.txt            # Python dependencies
├── enhanced_feature_list.json  # Feature configuration
└── README.md                   # This file
```

### 📦 Repository Size
- **Total size**: ~14 MB (well under GitHub's 100MB file limit)
- **Model file**: 832 KB
- **Historical data**: 13 MB (434,014 records from 2001-2025)

## 📝 Usage

1. **Select Date and Time**: Choose the date and time for prediction
2. **View Features**: Check derived features (season, weekend, peak hours)
3. **Predict**: Click the "Predict Demand" button
4. **Analyze**: View the prediction with visual gauge chart

## 🎓 Data Source

UK historic electricity demand data (2001-2025) from National Grid ESO.

## 🔬 Model Training

To retrain the model:

1. Ensure you have the historical data in `data/interim/elec_cleaned_full.parquet`
2. Open and run `02_models_local.ipynb` in Jupyter
3. The trained model will be saved to `data/final/models/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

**Student ID**: R0959568  
**Institution**: Thomas More University  
**Course**: Machine Learning - Cloud Deployment

## 🙏 Acknowledgments

- National Grid ESO for electricity demand data
- Thomas More University for project guidance
