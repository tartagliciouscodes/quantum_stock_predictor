# Quantum Stock Trend Predictor
## Project Documentation — Quantum Machine Learning Model

---

## 1. Project Title

**Quantum Stock Trend Predictor: Variational Quantum Classifier for Next-Day Stock Price Direction Prediction**

---

## 2. Abstract

This project implements a **4-qubit Variational Quantum Classifier (VQC)** using the PennyLane quantum machine learning framework to predict the next-day price direction (UP or DOWN) of Indian NSE-listed stocks. Four technical indicators — RSI, MACD, Daily Return, and Volatility — are extracted from historical OHLCV data and used as input features. The features are angle-encoded into quantum states using RY rotations, processed through two variational layers with RY, RZ, and CNOT gates, and a Pauli-Z expectation value on qubit 0 is used as the classification output. Training is performed using PennyLane's Adam optimizer minimising Binary Cross-Entropy loss. The model is evaluated on a strictly chronological out-of-sample test set of 323 samples. Test set accuracy is 49.23% with an F1 score of 48.75%. The model runs entirely on a classical simulation (default.qubit device) and does not claim quantum advantage.

---

## 3. Problem Statement

Stock market direction prediction is a binary classification problem: given a set of market observations up to day t, predict whether the closing price on day t+1 will be higher or lower than on day t.

This is a notoriously difficult problem due to:
- **Non-stationarity**: The statistical properties of financial time series change over time.
- **Noise**: Short-term price movements are heavily influenced by random market microstructure.
- **Complexity**: Many external factors (news, macroeconomics, sentiment) are not captured in price-based technical indicators alone.

---

## 4. Objective

1. Implement a technically accurate 4-qubit Variational Quantum Classifier using PennyLane.
2. Train and evaluate the VQC on historical stock market data with strict no-leakage chronological splits.
3. Deploy the model as an interactive, user-friendly web application using Streamlit.
4. Present an honest empirical evaluation of the model predictive capability.
5. Clearly communicate the methodology to both beginner and technical audiences.

---

## 5. Motivation

Quantum Machine Learning (QML) is an emerging field that explores whether quantum computing techniques can offer advantages for machine learning tasks. Variational Quantum Circuits are considered one of the most promising near-term QML approaches, as they can run on both quantum simulators and real quantum devices.

Financial time-series prediction represents a high-dimensional, noisy classification problem where quantum kernel methods and variational circuits have been theoretically proposed to offer advantages.

---

## 6. Dataset

**Source:** Yahoo Finance (primary) via the yfinance library, with a verified historical archive as fallback.

**Tickers supported:**
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- SBIN.NS (State Bank of India)
- ITC.NS (ITC Limited)

**Date range:** 2018-01-01 to present (live).
**Format:** Daily OHLCV data — Date, Open, High, Low, Close, Volume.

---

## 7. Data Preprocessing

Implemented in src/preprocessing.py — preprocess_data().

Steps performed:
1. Date conversion: pd.to_datetime() with timezone stripping.
2. Chronological sorting: Ascending sort by Date.
3. Duplicate removal: Drop duplicate rows based on Date (keep first).
4. Numeric conversion: pd.to_numeric(..., errors='coerce') for all OHLCV columns.
5. Invalid row removal: Drop rows where Open/High/Low/Close <= 0 or Volume < 0.
6. NaN removal: dropna() on all numeric columns.
7. Index reset: reset_index(drop=True).

---

## 8. Feature Engineering

Implemented in src/feature_engineering.py — create_features() using the ta library.

Features computed: SMA_10, SMA_20, SMA_50, EMA_20, RSI, MACD, MACD_Signal, BB_High, BB_Low, BB_Width, ATR, Daily_Return, Volatility, High_Low_Pct, Close_Open_Pct.

---

## 9. Four Quantum Features

The VQC uses only 4 features selected for complementary informational content:

### RSI — Relative Strength Index
- Source: ta.momentum.RSIIndicator(close=close, window=14).rsi()
- Measures: Price momentum over the last 14 days.
- Range: 0 to 100
- Interpretation: Above 70 = overbought; below 30 = oversold.

### MACD — Moving Average Convergence Divergence
- Source: ta.trend.MACD(close=close).macd()
- Measures: Difference between 12-day and 26-day exponential moving averages.
- Interpretation: Positive = bullish momentum; Negative = bearish momentum.

### Daily_Return — 1-Day Percentage Return
- Source: close.pct_change()
- Measures: Relative price change from the previous trading day.
- Interpretation: Positive = price went up; Negative = price went down.

### Volatility — 20-Day Rolling Volatility
- Source: Daily_Return.rolling(window=20).std()
- Measures: Standard deviation of daily returns over the past 20 trading days.
- Interpretation: Higher = greater price uncertainty.

---

## 10. Target Variable

Binary classification target (src/target_creation.py):
- 1 (UP): Next day's Close > Current day's Close
- 0 (DOWN): Next day's Close <= Current day's Close

Computed by shifting the Close column by -1 (forward-looking), then comparing. The final row is dropped. Zero data leakage.

---

## 11. Quantum Machine Learning Approach

Variational Quantum Classifier (VQC) — a hybrid quantum-classical algorithm:
1. Feature encoding: Map classical data into quantum states.
2. Variational ansatz: Apply trainable quantum gates.
3. Measurement: Extract a classical output for classification.

Implemented in src/quantum_model.py using PennyLane.

---

## 12. VQC Architecture

Device: qml.device('default.qubit', wires=4)
Weight shape: (n_layers=2, N_QUBITS=4, 2) = (2, 4, 2)
Total trainable parameters: 2x4x2=16 rotation angles + 1 bias = 17 parameters

Circuit structure:
- Feature encoding: RY(features[i], wires=i) for each qubit
- 2 variational layers, each containing:
  - RY(weights[l,i,0], wires=i) + RZ(weights[l,i,1], wires=i) for each qubit
  - Ring CNOT entanglement: CNOT([0,1], [1,2], [2,3], [3,0])
- Measurement: qml.expval(qml.PauliZ(0))
- Output: sigmoid(bias + 2.0 * expval) -> P(UP)

---

## 13. Qubit Encoding

Method: Angle Encoding (Rotation Encoding).

The 4 features [RSI, MACD, Daily_Return, Volatility] are first normalised using StandardScaler (fitted on training data only), then each feature is encoded as the rotation angle of an RY gate:

    for i in range(4):
        qml.RY(features[i], wires=i)

---

## 14. Quantum Gates

### Feature Encoding Gate
- qml.RY(angle, wires=i): Rotates qubit i by angle radians around the Y-axis of the Bloch sphere.

### Variational Gates (Trainable)
- qml.RY(weights[l, i, 0], wires=i): Parameterised Y-rotation in each variational layer.
- qml.RZ(weights[l, i, 1], wires=i): Parameterised Z-rotation in each variational layer.

### Entanglement Gate
- qml.CNOT(wires=[control, target]): Controlled-NOT gate. Flips the target qubit if and only if the control qubit is in state |1>.

---

## 15. Entanglement

Topology: Ring CNOT

    Q0 -> Q1
    Q1 -> Q2
    Q2 -> Q3
    Q3 -> Q0  (wraps around)

Code: qml.CNOT(wires=[i, (i + 1) % N_QUBITS]) for i in range(4)

Applied after single-qubit rotations in each variational layer.

---

## 16. Measurement

Operator: Pauli-Z expectation value on Qubit 0

    return qml.expval(qml.PauliZ(0))

Output range: <Z0> in [-1.0, +1.0]

Conversion to probability:
    prob = sigmoid(bias + 2.0 * expval)

Classification threshold: prediction = 1 (UP) if prob >= 0.5 else 0 (DOWN)

---

## 17. Training Process

### Train/Test Split
- Method: Strictly chronological, no shuffling
- Ratio: 80% train / 20% test
- Data leakage: Zero

### Feature Scaling
    scaler_q = StandardScaler()
    X_train_q_scaled = scaler_q.fit_transform(X_train_q)
    X_test_q_scaled = scaler_q.transform(X_test_q)

### Parameter Initialisation
    pnp.random.seed(42)
    weights = pnp.random.uniform(-pi, pi, shape=(2, 4, 2), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

### Optimisation
    opt = qml.AdamOptimizer(stepsize=0.05)

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | qml.AdamOptimizer(stepsize=0.05) |
| Loss | Binary Cross-Entropy (BCE) |
| Epochs | 15 |
| Batch size | 32 |
| Max train samples | 320 |
| Weight init | random.uniform(-pi, pi) seed=42 |
| Bias init | 0.0 |
| Threshold | 0.5 |

### Saved Artifacts
- models/quantum/quantum_vqc.pkl
- models/quantum/quantum_scaler.pkl
- models/quantum/results/quantum_circuit.png
- models/quantum/results/quantum_circuit.txt
- models/quantum/results/quantum_training_loss.png

---

## 18. Prediction Process

1. Download/load latest stock data.
2. Run preprocessing and feature engineering.
3. Load saved quantum model (quantum_vqc.pkl).
4. Load saved quantum scaler (quantum_scaler.pkl).
5. Extract 4 quantum features from the latest row.
6. Apply saved StandardScaler.transform().
7. Run quantum_circuit(weights, scaled_features) -> <Z0>.
8. Compute prob = sigmoid(bias + 2.0 * <Z0>).
9. Classify: prediction = 1 (UP) if prob >= 0.5 else 0 (DOWN).

---

## 19. Evaluation 


- Accuracy: Overall fraction of correct predictions
- Precision: Of all UP predictions, how many were correct
- Recall: Of all actual UP days, how many were correctly predicted
- F1 Score: Harmonic mean of Precision and Recall
- MCC: Matthews Correlation Coefficient — robust metric for imbalanced classes
- Confusion Matrix: Visual breakdown of TP, TN, FP, FN
- Pred UP/DOWN: Total count of each class (checks for class bias)

---

## 20. Results

Empirical results on chronological out-of-sample test set (RELIANCE.NS, 323 samples):

| Metric | Quantum VQC | Majority Class Baseline |
|--------|-------------|------------------------|
| Accuracy | 49.23% | 53.87% |
| Precision | 53.42% | 53.87% |
| Recall | 44.83% | 100.00% |
| F1 Score | 48.75% | 70.02% |
| MCC | -0.0081 | 0.0000 |
| Predicted UP | 146 | 323 |
| Predicted DOWN | 177 | 0 |

Key observations:
- The VQC accuracy (49.23%) is below the Majority Class Baseline (53.87%).
- MCC of -0.0081 confirms near-random classification performance.
- The VQC produced a balanced prediction distribution (146 UP / 177 DOWN), unlike classical tree-based models that were heavily biased toward DOWN.
- The VQC Recall (44.83%) is significantly better than Random Forest (4.60%) and XGBoost (5.75%).

HONEST CONCLUSION: The Quantum VQC does NOT demonstrate quantum advantage. Performance is near-random, consistent with the Efficient Market Hypothesis for technical indicator-based short-term prediction.

---

## 21. Limitations

1. Simulated quantum circuit: All computations use default.qubit (CPU simulator). No real quantum hardware.
2. Small feature set: Only 4 features used.
3. CPU performance: max_train_samples=320 limits training for practical execution time.
4. No quantum advantage demonstrated.
5. Technical indicators only: Macroeconomic and news factors not captured.
6. Single ticker training: No cross-ticker generalisation.
7. Hyperparameter sensitivity: VQC performance is sensitive to weight initialisation, layer count, and learning rate.

---

## 22. Future Improvements

1. More qubits (6-8) and deeper circuits (3-4 variational layers).
2. Alternative encoding strategies (amplitude encoding, IQP encoding).
3. Quantum kernel SVM as alternative to VQC.
4. GPU-accelerated simulation via PennyLane lightning plugin.
5. Additional technical indicator features.
6. News sentiment integration.
7. Testing on real quantum hardware (IBM Quantum, IonQ).
8. Hyperparameter search (Bayesian optimisation).

---

## 23. Conclusion

This project implements a fully functional, technically accurate 4-qubit Variational Quantum Classifier for stock market direction prediction using PennyLane. The model is trained on real historical OHLCV data from Yahoo Finance using four technical indicators (RSI, MACD, Daily Return, Volatility) as quantum features.

Empirical evaluation shows near-random predictive performance (49.23% accuracy, MCC = -0.0081), consistent with the Efficient Market Hypothesis.

The project demonstrates:
1. A correct and complete PennyLane VQC implementation (angle encoding, variational layers, ring CNOT, Pauli-Z measurement).
2. Honest, zero-leakage empirical evaluation methodology.
3. A user-friendly Streamlit application explaining quantum concepts to beginners while providing complete technical details for experts.

The project does NOT claim quantum advantage, does NOT use real quantum hardware, and does NOT provide financial advice.

---

## 24. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | >=3.9 | Core programming language |
| PennyLane | >=0.35.0 | Quantum ML framework — VQC implementation |
| NumPy | >=1.24.0 | Numerical computation |
| Pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.3.0 | StandardScaler, train/test utilities |
| yfinance | >=0.2.36 | Yahoo Finance data download |
| ta | >=0.11.0 | Technical indicator computation |
| Streamlit | >=1.31.0 | Web dashboard |
| Plotly | >=5.18.0 | Interactive charting |
| joblib | (bundled with sklearn) | Model serialisation |
| Matplotlib / Seaborn | (for artifact generation) | Circuit and loss diagrams |

---

## 25. How to Run the Project

### Prerequisites
    python -m venv venv
    venv\Scripts\activate        # Windows
    pip install -r requirements.txt

### Step 1 — Train the Quantum VQC
    python run.py --quantum --ticker RELIANCE.NS

This will:
- Download historical stock data from Yahoo Finance.
- Run preprocessing, feature engineering, and target creation.
- Train the 4-qubit VQC (15 epochs, Adam optimizer).
- Save model to models/quantum/quantum_vqc.pkl
- Save scaler to models/quantum/quantum_scaler.pkl
- Generate circuit diagram and training loss plots.

Note: Training on a CPU takes several minutes due to quantum simulation overhead.

### Step 2 — Launch the Dashboard
    streamlit run app/app.py

Open your browser at http://localhost:8501.

### Step 3 — Using the App
1. Select a stock ticker from the sidebar (or enter a custom one).
2. Click Analyze & Predict.
3. The prediction card shows UP or DOWN with a confidence percentage.
4. Scroll down to explore the 4 quantum feature signals, circuit diagram, and training loss curve.
5. Expand Technical Details for the full implementation specifications.

### Additional CLI Options
    python run.py --quantum --ticker TCS.NS --force-download
    python run.py --quantum --ticker INFY.NS --start-date 2020-01-01

---

This documentation describes the actual implementation in the repository. All technical values, parameters, and results are drawn directly from the source code and empirical evaluation — no values are invented.
