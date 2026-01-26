# Energy Fraud Detection System 🔍⚡

A comprehensive deep learning system for energy fraud detection using recurrent autoencoders and advanced classifiers. This system analyzes multiple time series data to identify anomalous patterns in energy consumption that may indicate fraud or system irregularities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This system combines recurrent autoencoders for feature extraction with advanced classifiers (DNN, XGBoost, and Random Forest) to identify anomalies in energy consumption patterns. The approach utilizes multiple time series for each energy meter and implements both threshold-based binary classification and supervised multi-class classification strategies.

### Key Features

- **Multi-modal Time Series Analysis**: Processes 5 different types of time series data per energy meter
- **Advanced Feature Extraction**: Uses recurrent autoencoders to generate meaningful embeddings
- **Multiple Classification Approaches**: Binary threshold-based and supervised multi-class classification
- **Ensemble Methods**: 4-fold cross-validation with hard/soft voting for robust predictions
- **Explainable AI**: SHAP analysis for feature importance and model interpretability
- **Flexible Architecture**: Modular design supporting different training strategies

## 📊 Dataset Structure

The system processes five types of time series for each Supply ID:

| Dataset Type | Description | Key Features |
|--------------|-------------|--------------|
| **ANAGRAFICA** | Customer details | Demographics, contract info, power ratings |
| **CONSUMI** | Quarter-hourly consumption | Energy usage patterns in kWh |
| **INTERRUZIONI** | Power interruptions | Outage events and duration |
| **LAVORI** | Maintenance works | Work activities on supply points |
| **PAROLE_DI_STATO** | Meter alarms | Triggered alerts and status messages |

### Target Classes

- **Class 0**: Anomalia (Anomaly)
- **Class 1**: Frode (Fraud)  
- **Class 2**: Regolare (Regular/Normal)

## 🏗️ System Architecture

```
Raw Data → Preprocessing → Autoencoders → Embeddings → Classification
    ↓           ↓             ↓            ↓             ↓
   CSV       Features    Compressed    Combined      Fraud/Anomaly
  Files      Engineering  Representations Vectors     Detection
```

### Core Components

1. **Data Preprocessing Pipeline** (`src/features/build_features.py`)
   - Missing value imputation
   - Temporal feature engineering with cyclic encoding
   - Categorical encoding and standardization
   - Sequence length normalization

2. **Recurrent Autoencoders** (`src/models/Autoencoder/`)
   - LSTM-based encoder-decoder architecture
   - 16-dimensional embedding generation
   - Reconstruction error computation for anomaly scoring

3. **Feature Fusion** (`src/embedding_generator.py`)
   - Intermediate-level fusion through concatenation
   - Combined embedding: `[e_anagrafica, e_consumi, e_interruzioni, e_lavori, e_parole, r_errors]`

4. **Classification Methods**
   - **Threshold-based Binary**: Adaptive threshold using reconstruction errors
   - **Supervised DNN**: Feed-forward neural network with dropout regularization
   - **XGBoost**: Gradient boosting with hyperparameter tuning
   - **Random Forest**: Ensemble of decision trees with class balancing

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install uv (recommended) or use pip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Davidermellino/Energy-Fraud-Detection-System-based-on-Deep-Learning.git
cd Energy-Fraud-Detection-System-based-on-Deep-Learning

```
2. **Download Data**
you can download the dataset from the [official challenge website](https://openinnovability.enel.com/it/challenge/call/2025/3/protezione-ricavi-energia-serie-temporali)


3. **Install dependencies**
```bash
# Using uv (recommended)
uv sync
```

4. **Prepare your data**
   - Place your CSV files in the `data/raw/` directory
   - Ensure files follow the expected naming convention:
     - `ANAGRAFICA.csv`
     - `CONSUMI.csv`
     - `INTERRUZIONI.csv`
     - `LAVORI.csv` 
     - `PAROLE_DI_STATO.csv`
     - `LABELS.csv`

### Basic Usage

#### 1. Run the Complete Pipeline

```bash
# Train autoencoders on regular data only (recommended)
python -m src.pipeline --mode XGBoost

# Available modes:
python -m src.pipeline --mode DNN_Classifier
python -m src.pipeline --mode RandomForest
python -m src.pipeline --mode Binary_Classifier  # Only works with ONLY_REGULAR=True
```

#### 2. Configuration

Edit `src/models/config.py` to customize training parameters:

```python
class Config:
    # Data settings
    DATASET_PATH = "data/processed/"
    ONLY_REGULAR = True  # Train autoencoders on regular samples only
    
    # Model parameters
    EMBEDDING_DIM = 16
    LEARNING_RATE_DNN = 0.001
    LEARNING_RATE_XG = 0.0005
    MAX_DEPTH = 4
    
    # Training settings
    N_EPOCHS_AE = 100
    N_EPOCHS_DNN = 200
    BATCH_SIZE = 32
    TEST_SIZE = 0.4
    SEED = 42
```


## 📈 Performance Metrics

### Comparison Results (Only Regular Mode)

| Method | Weighted Recall | Accuracy | F1-Score (Fraud) |
|--------|----------------|----------|------------------|
| **DNN Soft Voting** | **0.568** | 0.73 | **0.29** |
| DNN Hard Voting | 0.560 | 0.70 | 0.27 |
| Threshold Binary | 0.446 | 0.81 | 0.296 |
| XGBoost | 0.026 | 0.85 | 0.00 |
| Random Forest | 0.027 | 0.90 | 0.00 |

### Key Findings

- **DNN outperforms** other methods in minority class detection (crucial for fraud detection)
- **"Only Regular" training** significantly improves performance over "All Clusters" approach
- **Soft voting** consistently beats hard voting in ensemble methods
- **Trade-off**: Higher accuracy doesn't always mean better fraud detection capability

## 🔍 Feature Importance Analysis

SHAP analysis reveals dataset contributions to model decisions:

### DNN Classifier (Only Regular)
- **INTERRUZIONI**: 29.9% - Power interruption patterns are highly discriminative
- **ANAGRAFICA**: 25.2% - Customer demographics and contract details
- **CONSUMI**: 20.7% - Energy consumption patterns
- **LAVORI**: 13.8% - Maintenance work history
- **PAROLE_DI_STATO**: 10.4% - Alarm and status messages

## 📁 Project Structure

```
deep_learning_for_energy_recovery/
├── data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Preprocessed data
│   ├── embeddings/             # Generated embeddings
│   └── combined_embeddings/    # Fused feature vectors
├── models/
│   ├── Autoencoder/           # Trained autoencoder models
│   ├── DNN/                   # Neural network models
│   ├── XGBoost/               # XGBoost models
│   └── RandomForest/          # Random Forest models
├── src/
│   ├── features/
│   │   └── build_features.py  # Data preprocessing
│   ├── models/
│   │   ├── config.py          # Configuration settings
│   │   ├── trainers.py        # Model training classes
│   │   └── combine.py         # Model combination utilities
│   ├── data_processing.py     # Data loading and sequence processing
│   ├── embedding_generator.py # Embedding generation
│   ├── binary_classifiers.py  # Threshold-based classification
│   ├── evaluation.py          # Model evaluation and metrics
│   ├── training_orchestrator.py # Training coordination
│   └── pipeline.py            # Main pipeline orchestration
├── notebooks/
│   └── 1.0-de-initial-data-exploration.ipynb
└── references/
    └── 2503.13709v1.pdf       # Research paper reference
```


### Training Modes

1. **"Only Regular" Mode** (Recommended)
   ```python
   Config.ONLY_REGULAR = True
   ```
   - Trains autoencoders exclusively on normal samples
   - Better anomaly detection capability
   - Higher sensitivity to irregular patterns

2. **"All Clusters" Mode**
   ```python
   Config.ONLY_REGULAR = False
   ```
   - Uses all available data for training
   - May reduce generalization for anomaly detection
   - Useful for baseline comparisons

you can edit the configuration in `src/models/config.py` to switch modes and adjust parameters.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Davide Ermellino**  
Università degli Studi di Cagliari

## 🔗 References

- [Multimodal Time Series Analysis Paper](references/2503.13709v1.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
