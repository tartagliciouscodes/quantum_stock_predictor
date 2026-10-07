"""
Validation script for End-to-End Pipeline testing of RELIANCE.NS and TCS.NS.
"""
import os
import sys

# Set stdout encoding for Windows console unicode emoji printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.prediction import get_latest_stock_features, predict_trend, predict_all_models, QUANTUM_FEATURE_NAMES

def validate_ticker(ticker: str):
    print(f"\n==========================================")
    print(f"VALIDATING END-TO-END PIPELINE: {ticker}")
    print(f"==========================================")
    
    # 1. Fetch features
    featured_df, overview = get_latest_stock_features(ticker)
    print(f"Data retrieved successfully. Total rows: {len(featured_df)}")
    print(f"Latest Trading Date: {overview['latest_date']}")
    print(f"Latest Close Price: {overview['latest_close']}")
    print(f"Daily Change: {overview['daily_change']} ({overview['daily_pct']}%)")
    print(f"Indicators - RSI: {overview['rsi']}, MACD: {overview['macd']}, Volatility: {overview['volatility']}")
    
    # 2. Test individual model predictions
    models = ["Logistic Regression", "Random Forest", "XGBoost", "Quantum VQC"]
    for m in models:
        res = predict_trend(ticker=ticker, model_name=m, featured_df=featured_df)
        prob_str = f"{res['predicted_probability']*100:.2f}%" if res['predicted_probability'] is not None else "N/A"
        print(f"Model: {m:20s} | Prediction: {res['prediction_label']} | Prob: {prob_str}")
        assert res['prediction'] in [0, 1], f"Invalid prediction value for {m}"
        assert res['latest_close_price'] > 0, "Invalid close price"
        assert res['latest_trading_date'] == overview['latest_date'], "Date mismatch"

    # 3. Test predict_all_models
    all_res = predict_all_models(ticker=ticker, featured_df=featured_df)
    assert len(all_res) == 4, "predict_all_models should return 4 results"
    print(f"All-model batch prediction successful for {ticker}.")

if __name__ == "__main__":
    print(f"Quantum Feature Set: {QUANTUM_FEATURE_NAMES}")
    assert QUANTUM_FEATURE_NAMES == ['RSI', 'MACD', 'Daily_Return', 'Volatility'], f"Quantum features mismatch: {QUANTUM_FEATURE_NAMES}"
    print("Quantum features verified: RSI, MACD, Daily_Return, Volatility")
    
    validate_ticker("RELIANCE.NS")
    validate_ticker("TCS.NS")
    print("\nALL END-TO-END INFERENCE TESTS PASSED SUCCESSFULLY!")
