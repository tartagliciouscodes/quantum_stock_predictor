# Hybrid Quantum-Classical AI-Based Stock Market Trend Prediction System

A capstone project comparing Classical Machine Learning algorithms (Logistic Regression, Random Forest, XGBoost) against a Quantum Machine Learning (QML) Variational Quantum Classifier (VQC) implemented using PennyLane for short-term binary stock market trend prediction (UP 📈 / DOWN 📉).

---

## 📌 Abstract

Short-term stock market prediction is a challenging problem in financial computing due to high volatility and dynamic market dynamics. This project presents a hybrid quantum-classical artificial intelligence architecture designed to classify next-day stock price trends as **UP** or **DOWN**.

> [!IMPORTANT]
> **Real Market Data Policy:**
> - This project relies **exclusively on real historical market data** sourced from Yahoo Finance.
> - **Synthetic / fake generated stock data is strictly prohibited** and is never used for model training, testing, or evaluation.
> - If live API endpoints are restricted by network policies or SSL connection resets, verified real historical market datasets are loaded from local CSV cache or historical archives.
> - Users can also manually supply a standard historical OHLCV CSV file at `data/raw/{ticker}.csv`.


---

## 🎯 Problem Statement

Determining short-term stock price trends using continuous numerical target forecasting often yields high noise and overfitting. Frame the problem as a **binary trend classification**:
- **Target = 1 (UP):** Tomorrow's Close Price > Today's Close Price
- **Target = 0 (DOWN):** Tomorrow's Close Price ≤ Today's Close Price

The main research question is:
> **“Can a Quantum Machine Learning model perform competitively with classical machine-learning models for short-term stock market trend prediction?”**

## 🤖 Classical Machine Learning Baselines

The system implements three classical machine learning baselines trained on the 20-feature dataset:
1. **Logistic Regression:** Linear decision boundary baseline with L2 regularization (`C=1.0`). Uses scaled features.
2. **Random Forest:** Ensemble of decision trees (`n_estimators=100`, `max_depth=6`, `min_samples_leaf=5`) capturing non-linear feature interactions.
3. **XGBoost:** Gradient boosted decision trees (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`).

### Chronological Train/Test Split (Zero Data Leakage)
- **Training Set (80%):** 1,291 samples (`2018-03-14` -> `2023-06-05`)
- **Testing Set (20%):** 323 samples (`2023-06-06` -> `2024-09-26`)
- `StandardScaler` is fitted **strictly on training data** and applied to test data.

---

## ⚛️ Quantum Machine Learning (Phase 10)

### 1. Quantum Feature Selection (4 Features)
To comply with NISQ-era simulation limitations on classical computers, exactly 4 core normalized features were selected:
1. **`RSI`** (Relative Strength Index - Momentum)
2. **`MACD`** (Moving Average Convergence Divergence - Trend)
3. **`Daily_Return`** (1-Day Price Return - Velocity)
4. **`Volatility`** (20-Day Rolling Return Volatility - Risk)

All features are standardized via `StandardScaler` fitted **strictly on the 1,291 training samples** and transformed without data leakage.

### 2. Quantum Circuit Architecture
- **Simulator:** PennyLane `default.qubit` (4 Qubits)
- **Feature Encoding:** `qml.RY(x_i, wires=i)` Angle Encoding mapping classical features directly to single-qubit state rotations.
- **Variational Layers (2 Layers):** Parameterized single-qubit rotations (`qml.RY(w_1)`, `qml.RZ(w_2)`) with ring CNOT entanglement:
  ```
  Q0 ──RY(x0)──RY(w0)──RZ(w1)──●─────────────X── ⟨Z0⟩
                               │             │
  Q1 ──RY(x1)──RY(w2)──RZ(w3)──X──●──────────┼──
                                  │          │
  Q2 ──RY(x2)──RY(w4)──RZ(w5)─────X──●───────┼──
                                     │       │
  Q3 ──RY(x3)──RY(w6)──RZ(w7)────────X───────●──
  ```
- **Measurement & Classification:** Expectation value $\langle Z_0 \rangle$ of the Pauli-Z operator on Qubit 0, mapped to class probability via $\sigma(\text{bias} + 2 \cdot \langle Z_0 \rangle)$.

---

## 🔬 Phase 11 — Experiment Analysis & Model Comparison

Below are the **real empirical test set evaluation results** generated from actual execution on the `RELIANCE.NS` chronological test dataset (`323` samples, `2023-06-06` $\rightarrow$ `2024-09-26`):

### 1. Test Dataset Class Distribution
- **Actual UP (`Target = 1`):** **174 samples (53.87%)**
- **Actual DOWN (`Target = 0`):** **149 samples (46.13%)**

### 2. Comprehensive Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | MCC | Pred UP | Pred DOWN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (Always UP)** | **0.5387** | 0.5387 | 1.0000 | 0.7002 | 0.0000 | 323 | 0 |
| **Random Baseline (50/50)** | 0.5077 | 0.5466 | 0.5057 | 0.5254 | 0.0158 | 161 | 162 |
| **Logistic Regression** | 0.4644 | 1.0000 | 0.0057 | 0.0114 | 0.0516 | 1 | 322 |
| **Random Forest** | 0.4768 | 0.7273 | 0.0460 | 0.0865 | 0.0710 | 11 | 312 |
| **XGBoost** | 0.4675 | 0.5556 | 0.0575 | 0.1042 | 0.0082 | 18 | 305 |
| **Quantum VQC (PennyLane)** | 0.4923 | 0.5342 | 0.4483 | 0.4875 | -0.0081 | 146 | 177 |

---

## 🧐 Detailed Model Performance Interpretation

1. **Naive Baselines:**
   - The **Majority Class Baseline** (always predicting UP) achieves **53.87% accuracy**, setting the simple benchmark for this test period.
   - The **Uniform Random Baseline** achieves **50.77% accuracy** and an MCC near zero (`0.0158`).

2. **Classical ML Models (Logistic Regression, Random Forest, XGBoost):**
   - **Accuracy Range:** 46.44% – 47.68%.
   - **Prediction Bias:** All three classical models exhibit extreme prediction bias toward class `0` (DOWN). Logistic Regression predicted 322 DOWN vs 1 UP; Random Forest predicted 312 DOWN vs 11 UP; XGBoost predicted 305 DOWN vs 18 UP.
   - **Cause:** Standard probability calibration thresholds (0.50) without class-weight rebalancing cause these models to default to predicting DOWN during out-of-sample regime shifts, resulting in near-zero recall (0.0057 – 0.0575).

3. **Quantum VQC (PennyLane):**
   - **Accuracy:** 49.23% (matching the random baseline range).
   - **Balanced Prediction Profile:** Unlike classical tree models, the VQC predicted **146 UP and 177 DOWN**, yielding a balanced recall profile (0.4483) and F1 score (0.4875).
   - **Scientific Caution:** The VQC produced a more balanced recall profile than the classical models on this test period. **This observation does NOT establish that quantum superposition or entanglement caused the difference.** The MCC of -0.0081 confirms performance remains near random classification.

---

## 🚫 Threats to Validity & Limitations

1. **Single-Stock Scope:** Experiments are evaluated on a single asset (`RELIANCE.NS`).
2. **Limited Historical Period:** Evaluated over 2018–2024 time-series split (`1,614` total records).
3. **No Quantum Advantage:** Empirical results demonstrate **no quantum advantage or supremacy**. The Quantum VQC performs competitively with classical baselines but does not beat the simple Majority Class Baseline (53.87%).
4. **Simulator vs Hardware:** Execution was performed on a classical statevector simulator (`default.qubit`), not physical NISQ hardware.
5. **No Profitability / Financial Trading Claims:** Accuracy near 50% does not constitute a profitable trading strategy. No backtesting or transaction-cost modeling is claimed.



### Top 10 Feature Importances (Random Forest)
1. `BB_Width` (0.0856)
2. `High_Low_Pct` (0.0674)
3. `SMA_10` (0.0626)
4. `RSI` (0.0608)
5. `SMA_50` (0.0573)
6. `Volume` (0.0550)
7. `Daily_Return` (0.0515)
8. `SMA_20` (0.0511)
9. `MACD_Signal` (0.0486)
10. `Close_Open_Pct` (0.0480)

### Top 10 Feature Importances (XGBoost)
1. `SMA_10` (0.0645)
2. `EMA_20` (0.0543)
3. `Daily_Return` (0.0536)
4. `BB_Width` (0.0527)
5. `Volume` (0.0517)
6. `RSI` (0.0512)
7. `MACD_Signal` (0.0509)
8. `Low` (0.0504)
9. `Close_Open_Pct` (0.0504)
10. `ATR` (0.0503)

---

## 🏆 Objectives

1. Develop a modular, reproducible Python application for automated financial data collection and preprocessing.
2. Calculate robust technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) without data leakage.
3. Train classical ML benchmarks (Logistic Regression, Random Forest, XGBoost).
4. Construct and train a 4-qubit Quantum Neural Network (QNN) using PennyLane on a classical simulator (`default.qubit`).
5. Compare model performance transparently using Accuracy, Precision, Recall, F1 Score, and Confusion Matrices.
6. Provide an interactive Streamlit dashboard for real-time visualization, model training, and stock trend inference.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────┐
                               │ Yahoo Finance (yfinance)│
                               └───────────┬─────────────┘
                                           │ Historical OHLCV
                                           ▼
                               ┌─────────────────────────┐
                               │ Data Preprocessing &    │
                               │ Feature Engineering     │
                               └───────────┬─────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │   Full Feature Set    │                     │  4 Quantum Features   │
        │ (SMA, EMA, RSI, MACD) │                     │(RSI, MACD, Ret, Vol)  │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │ Classical ML Models   │                     │  Angle Encoding &     │
        │ (Logistic, RF, XGB)   │                     │ PennyLane 4-Qubit VQC │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │ Performance Metrics
                                           ▼
                               ┌─────────────────────────┐
                               │  Streamlit Web App &    │
                               │   Evaluation Engine     │
                               └─────────────────────────┘
```

---

## 💻 Tech Stack

- **Language:** Python 3.10+
- **Data Collection:** `yfinance`
- **Data Manipulation:** `pandas`, `numpy`
- **Technical Indicators:** `ta`
- **Classical ML:** `scikit-learn`, `xgboost`
- **Quantum Machine Learning:** `pennylane` (using `default.qubit` simulator backend)
- **Frontend / Visualization:** `streamlit`, `plotly`
- **Testing:** `pytest`

---

## 📁 Project Structure

```
Quantum_Stock_Predictor/
│
├── app/
│   └── app.py                     # Streamlit dashboard application
│
├── data/
│   ├── raw/                       # Cached raw stock data CSVs
│   └── processed/                 # Cleaned datasets ready for training
│
├── models/
│   ├── classical/                 # Saved classical ML model artifacts
│   └── quantum/                   # Saved quantum weights & scalers
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py         # Yahoo Finance download & caching engine
│   ├── preprocessing.py           # Missing value handling & date sorting
│   ├── feature_engineering.py     # Technical indicator computation
│   ├── target_creation.py         # Shift(-1) binary target creation
│   ├── classical_models.py        # Logistic Regression, RF, XGBoost training
│   ├── quantum_model.py           # PennyLane 4-qubit VQC model
│   ├── evaluation.py              # Metric calculation & confusion matrix
│   └── prediction.py              # Real-time inference pipeline
│
├── notebooks/                     # Exploratory Data Analysis & experiments
├── tests/
│   ├── __init__.py
│   └── test_data_pipeline.py      # Automated pytest suite
│
├── requirements.txt               # Dependency specifications
├── README.md                      # Comprehensive documentation
└── run.py                         # Pipeline execution CLI
```

---

## 📊 Dataset & Supported Tickers

Historical OHLCV data is retrieved from Yahoo Finance starting from **2018-01-01** to the latest available trading date.

Supported initial tickers:
- `RELIANCE.NS` (Reliance Industries - Default)
- `TCS.NS` (Tata Consultancy Services)
- `INFY.NS` (Infosys)
- `HDFCBANK.NS` (HDFC Bank)
- `SBIN.NS` (State Bank of India)
- `ITC.NS` (ITC Limited)

---

## ⚙️ Feature Engineering & Selection

### Full Classical Feature Set (20 Input Features)
1. **Price Features:** `Open`, `High`, `Low`, `Close`, `Volume`
2. **Moving Averages:**
   - `SMA_10`, `SMA_20`, `SMA_50`: Simple Moving Averages computed over 10, 20, and 50 trading days to capture short-term and medium-term price trends.
   - `EMA_20`: Exponential Moving Average placing higher weight on recent price observations.
3. **Momentum Indicators:**
   - `RSI` (14-day Relative Strength Index): Measures velocity and magnitude of directional price movements (0–100 oscillator).
   - `MACD` & `MACD_Signal`: Moving Average Convergence Divergence line and signal line capturing trend reversals.
4. **Volatility & Risk Indicators:**
   - `BB_High`, `BB_Low`, `BB_Width`: 20-day Bollinger Bands (Upper band, Lower band, and Bandwidth) measuring price dispersion.
   - `ATR` (14-day Average True Range): Quantifies absolute market volatility.
   - `Daily_Return`: Percentage change in closing price relative to previous trading day (`Close.pct_change()`).
   - `Volatility`: 20-day rolling standard deviation of daily returns.
5. **Percentage Features:**
   - `High_Low_Pct`: Intraday price range percentage `(High - Low) / Low * 100`.
   - `Close_Open_Pct`: Daily candlestick body change percentage `(Close - Open) / Open * 100`.

### Selected Quantum Features (4 Qubits):
1. **RSI** (Relative Strength Index)
2. **MACD** (Moving Average Convergence Divergence)
3. **Daily Return**
4. **Volatility**

---

## 🎯 Target Definition & Data Leakage Prevention

### Binary Target Definition
The target variable is defined as a binary classification decision:
```text
If Tomorrow's Close > Today's Close:
    Target = 1 (UP 📈)
Otherwise:
    Target = 0 (DOWN 📉)
```

Constructed in code via: `Target = (Close.shift(-1) > Close).astype(int)`

### Strict Data Leakage Prevention
- **Future Information Isolation:** Tomorrow's closing price is used **exclusively** to calculate the ground-truth binary `Target` label.
- **Feature Cleanliness:** Tomorrow's price is immediately discarded after label generation and **NEVER** included in the feature matrix `X`.
- **Trailing Row Removal:** The final row of the dataset is automatically removed because tomorrow's closing price does not yet exist.
- **Sequential Indicator Windowing:** Technical indicators rely strictly on historical price lookbacks (`t-1`, `t-2`, ...) and do not look forward.


---

## ⚛️ Quantum Machine Learning Architecture

The Quantum Classifier is structured as a **Variational Quantum Classifier (VQC)**:
1. **Input Encoding:** 4 classical feature inputs normalized and encoded into quantum state amplitudes/angles using `qml.AngleEmbedding` or `qml.IQPEmbedding`.
2. **Variational Circuit:** Parametrized single-qubit rotations (`qml.Rot` / `qml.RY` / `qml.RZ`) parameterized by trainable weight matrix $\theta$.
3. **Entanglement Layer:** Controlled-NOT (`CNOT`) entangling gates connecting adjacent qubits to capture inter-feature non-linear correlations.
4. **Measurement:** Expectation value of Pauli-Z operator $\langle Z_0 \rangle$ mapped through a sigmoid / bias layer to produce class probability $P(Y=1)$.
5. **Optimization:** Trained using Adam or gradient descent on Binary Cross-Entropy loss.

---

## 🚀 Installation & Running

### 1. Environment Setup

Clone/navigate to project directory and install dependencies:
```bash
cd C:\Users\prart\.gemini\antigravity-ide\scratch\Quantum_Stock_Predictor
pip install -r requirements.txt
```

### 2. Run Data Collection & Preprocessing CLI (Phase 1-3)

```bash
python run.py --ticker RELIANCE.NS
```

### 3. Run Test Suite

```bash
pytest tests/
```

### 4. Launch Streamlit Web Application (Phase 13)

```bash
streamlit run app/app.py
```

---

## ⚠️ Limitations

- **Market Unpredictability:** Stock price movements are influenced by macroeconomic factors, news, and sentiment not fully captured in technical indicators alone.
- **Quantum Simulation Scale:** Simulating quantum circuits on classical computers scales exponentially with qubit count. The circuit depth and qubit count (4 qubits) are kept small for execution efficiency.
- **No Quantum Advantage Guarantee:** QML models on current NISQ simulators do not inherently outperform well-tuned classical algorithms like XGBoost.

---

## 🔮 Future Enhancements

- Integration of Sentiment Analysis from financial news APIs.
- Exploration of Hybrid Quantum-LSTM / Quantum Recurrent Architectures.
- Execution on actual Quantum Hardware via IBM Quantum / Amazon Braket plugins.
- Portfolio-level multi-asset quantum trend optimization.
