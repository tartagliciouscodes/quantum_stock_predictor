"""
Audit script for Final Data Integrity Audit.
"""
import os
import glob
import pandas as pd

def audit_files():
    raw_files = sorted(glob.glob("data/raw/*.csv"))
    processed_files = sorted(glob.glob("data/processed/*.csv"))

    print("==========================================")
    print("1. RAW DATA FILES AUDIT")
    print("==========================================")
    raw_data_summary = {}
    for f in raw_files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'])
        fname = os.path.basename(f)
        min_d = df['Date'].min().strftime('%Y-%m-%d')
        max_d = df['Date'].max().strftime('%Y-%m-%d')
        raw_data_summary[fname] = {
            'rows': len(df),
            'min_date': min_d,
            'max_date': max_d,
            'first_close': df['Close'].iloc[0],
            'last_close': df['Close'].iloc[-1],
            'mean_close': df['Close'].mean()
        }
        print(f"File: {fname:20s} | Rows: {len(df):5d} | Date Range: {min_d} to {max_d} | First Close: {df['Close'].iloc[0]:.2f} | Last Close: {df['Close'].iloc[-1]:.2f}")

    print("\n==========================================")
    print("2. PROCESSED DATA FILES AUDIT")
    print("==========================================")
    for f in processed_files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'])
        fname = os.path.basename(f)
        min_d = df['Date'].min().strftime('%Y-%m-%d')
        max_d = df['Date'].max().strftime('%Y-%m-%d')
        print(f"File: {fname:30s} | Rows: {len(df):5d} | Date Range: {min_d} to {max_d}")

    # Check distinctness of RELIANCE.NS vs TCS.NS
    if 'RELIANCE.NS.csv' in raw_data_summary and 'TCS.NS.csv' in raw_data_summary:
        rel = raw_data_summary['RELIANCE.NS.csv']
        tcs = raw_data_summary['TCS.NS.csv']
        print("\n==========================================")
        print("3. TICKER DISTINCTNESS CHECK (RELIANCE vs TCS)")
        print("==========================================")
        print(f"RELIANCE.NS: Last Close = {rel['last_close']:.2f}, Mean Close = {rel['mean_close']:.2f}")
        print(f"TCS.NS:      Last Close = {tcs['last_close']:.2f}, Mean Close = {tcs['mean_close']:.2f}")
        is_distinct = (rel['last_close'] != tcs['last_close']) and (rel['mean_close'] != tcs['mean_close'])
        print(f"Distinct Ticker Data Confirmed: {is_distinct}")
        assert is_distinct, "TCS.NS and RELIANCE.NS must have distinct price data!"

    # Check other tickers
    all_supported = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
    print("\n==========================================")
    print("4. TICKER LOCAL DATA AVAILABILITY CHECK")
    print("==========================================")
    for ticker in all_supported:
        raw_p = os.path.join("data", "raw", f"{ticker}.csv")
        exists = os.path.exists(raw_p)
        print(f"Ticker: {ticker:15s} | Local Cache Exists: {exists}")

if __name__ == "__main__":
    audit_files()
