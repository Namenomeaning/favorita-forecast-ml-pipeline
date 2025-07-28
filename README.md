# Store Sales Forecasting Project

Time series forecasting for Favorita stores in Ecuador using AutoGluon, Chronos, and deep learning. Includes data processing, feature engineering, and robust model training.

## 📊 Project Overview

Forecasts daily sales for thousands of product families at 54 Favorita stores. Uses:
- 54 stores, 33 product families
- External data: oil prices, holidays, transactions, promotions
- Models: Chronos, TemporalFusionTransformer, DeepAR

## 🏗️ Project Structure

```
favorita-forecast/
├── data/
│   ├── csv/                # Raw CSV files
│   └── featured/           # Processed data for training
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── process.py          # Main pipeline
│   └── data/               # Data modules
│       ├── load_data.py
│       ├── preprocess_data.py
│       ├── clean_data.py
│       ├── feature_engineering.py
│       ├── merge_data.py
│       └── save_data.py
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## 🚀 Quick Start

### 1. Installation

Clone the repository and enter the folder:
```bash
git clone <repository-url>
cd favorita-forecast
```

Install dependencies (choose one):
```bash
# With pip
pip install -r requirements.txt
# Or with uv (recommended)
uv pip install -r requirements.txt
```

For GPU support (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Setup
Place raw CSV files in `data/csv/`:
- train.csv, test.csv, stores.csv, transactions.csv, oil.csv, holidays_events.csv

### 3. Data Processing
Run the pipeline:
```bash
python src/process.py
# Or with uv
uv pip run python src/process.py
```
Processed data will be saved to `data/featured/`.

### 4. Model Training
Open and run:
```bash
jupyter notebook notebooks/machine_learning.ipynb
```

## 🔧 Features

### Data Processing
- Oil price interpolation
- Holiday/event handling
- Store activity filtering
- Zero-sales handling
- Time & lag features

### Model Capabilities
- Chronos, TFT, DeepAR
- Ensemble methods
- Probabilistic forecasting
- Covariate support

### Evaluation
- MASE metric
- Time-aware cross-validation
- Forecast visualization
- Performance tracking

## 📈 Current Results

### ✅ Progress Update
- Training pipeline complete
- Baseline comparison (ARIMA/Prophet) implemented
- Latest MASE score: **0.9** (target achieved)

### ⚙️ Technical Implementation
Applied deep **feature engineering** with key components:
- **Historical sales patterns**
- **Oil price indicators** 
- **Calendar features** (day of week, holidays)
- **Holiday flags** for special events
- **Advanced visualization** to understand interaction patterns

### ⚠️ Current Issues
- **Zero-sales items**: Affect prediction errors and MASE scores
- **Large data files**: Managed via `.gitignore` and local storage

### 🔄 Next Steps
Immediate focus:
- Zero-sales analysis
- Model optimization (target MASE < 0.8)
- Documentation
- Monthly retraining pipeline design

## 🎯 Project Status & Roadmap

### ✅ Completed
- Data pipeline (ETL)
- Feature engineering
- Model training (AutoGluon, Chronos, TFT, DeepAR)
- Evaluation framework (MASE, visualization)
- Prediction pipeline

### 🚧 Next Phase
1. **Model optimization** (hyperparameters, selection, zero-sales, ensemble)
2. **MLOps pipeline** (monthly ingestion, drift detection, retraining, versioning, A/B testing)
3. **Deployment & monitoring** (API, real-time, dashboards, alerts, CI/CD)
4. **Advanced features** (multi-horizon, hierarchical, external data, causal inference)

### 📋 Immediate Next Steps
1. Zero-sales analysis
2. Baseline comparison (ARIMA/Prophet)
3. Model optimization (MASE < 0.8)
4. Documentation
5. Monthly retraining pipeline

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or suggestions, please open an issue or contact the development team.

---
*Last updated: July 28, 2025*