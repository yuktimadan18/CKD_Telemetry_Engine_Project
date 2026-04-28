# 🏥 CKD Telemetry Engine

A real-time clinical decision support system for **Chronic Kidney Disease (CKD) stage prediction** using machine learning. The system ingests live patient telemetry data, computes trajectory-aware features, and generates stage-wise risk predictions with clinical alerts.

## ✨ Key Features

- **Multi-class CKD prediction** across 5 stages (Healthy → Stage 5 Kidney Failure)
- **6 ML models benchmarked**: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, Neural Network
- **Primary model**: Random Forest Classifier — **93.2% accuracy, 0.993 ROC-AUC**
- **Real-time streaming** simulation with health drift and crisis events
- **Trajectory-aware features**: eGFR velocity & acceleration via sliding window
- **Clinical alerting**: Automated STABLE / WARNING / CRITICAL alerts
- **Streamlit dashboard**: Live monitoring with metrics, charts, and probability distributions
- **SHAP explainability**: Feature impact analysis for model interpretability

## 📊 Results Summary

| Model | Accuracy | F1 (Weighted) | ROC-AUC |
|-------|----------|---------------|---------|
| K-Nearest Neighbors | 0.8627 | 0.8480 | 0.8806 |
| Logistic Regression | 0.8720 | 0.8562 | 0.9813 |
| Neural Network (MLP) | 0.9107 | 0.9045 | 0.9789 |
| Support Vector Machine | 0.9280 | 0.9226 | 0.9926 |
| Gradient Boosting | 0.9280 | 0.9266 | 0.9886 |
| **Random Forest** | **0.9320** | **0.9288** | **0.9933** |

## 🏗️ Architecture

```
CSV Dataset → Stream Generator → Sliding Window Features → Random Forest → Clinical Alert
                                       ↓
                              eGFR Velocity & Acceleration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ckd_telemetry_engine
pip install -r requirements.txt
```

### 2. Train Models & Generate Results

```bash
python research_pipeline.py
```

This trains all 6 models, generates comparison tables, confusion matrices, ROC curves, feature importance plots, SHAP analysis, and saves the trained Random Forest model.

### 3. Run Real-Time Telemetry Engine

```bash
python main.py
```

Starts the live telemetry stream with real-time CKD predictions. Press `Ctrl+C` to stop.

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

Opens a live Streamlit dashboard at `http://localhost:8501` with real-time monitoring.

## 📁 Project Structure

```
ckd_telemetry_engine/
├── main.py                 # Real-time telemetry engine
├── research_pipeline.py    # 6-model ML benchmark pipeline
├── dashboard.py            # Streamlit live dashboard
├── requirements.txt        # Python dependencies
├── data/
│   └── Testing_CKD_dataset.csv   # 4800-patient CKD dataset
├── models/                 # Trained model artifacts (generated)
├── results/                # Charts, tables, reports (generated)
└── src/
    ├── stream.py           # Patient telemetry stream generator
    ├── features.py         # Sliding-window feature engineering
    ├── inference.py        # ML model loading & prediction
    └── alert.py            # Clinical alert translation
```

## 🔬 Methodology

- **Dataset**: 4,800 patient records with 35 clinical features
- **Preprocessing**: Leaky feature removal, Gaussian noise injection, stratified splitting
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-fold cross-validation
- **Explainability**: SHAP feature attribution for clinical interpretability

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML | scikit-learn |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Explainability | SHAP |
| Dashboard | Streamlit |
