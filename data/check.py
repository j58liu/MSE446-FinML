import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
START_DATE = '2016-01-01'
END_DATE = '2025-12-31'
TARGET_TICKER = 'XLE'

# Macro and Geopolitical Features
MACRO_TICKERS = {
    '^VIX': 'VIX_Fear_Index',
    '^OVX': 'Oil_Volatility',
    'CL=F': 'Crude_Oil_Futures',
    'NG=F': 'Natural_Gas_Futures',
    'DX-Y.NYB': 'US_Dollar_Index_DXY',
    'ES=F': 'SP500_Futures',
    '^N225': 'Nikkei_225',
    '^FTSE': 'FTSE_100'
}

# The text file containing your tickers (one per line)
TICKERS_FILE = 'data/energy_tickers.txt'

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_tickers_from_file(filepath):
    """Reads tickers from a text file, handling blank lines and whitespace."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find '{filepath}'. Ensure it is in the same folder as this script.")
        
    with open(filepath, 'r') as file:
        # Strip whitespace, ignore empty lines, and uppercase everything
        tickers = [line.strip().upper() for line in file if line.strip()]
        
    print(f"Successfully loaded {len(tickers)} tickers from {filepath}.")
    return list(set(tickers))  # Remove any accidental duplicates

def filter_valid_stocks(tickers, start_year=2018, min_price=5.0):
    """Filters stocks based on trading history, minimum price, and active trading status."""
    valid_tickers = []
    print(f"Filtering {len(tickers)} Finviz tickers...")
    
    # Get today's date to check if the stock is still trading
    now = pd.Timestamp.now()
    
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="max")
            
            if hist.empty:
                continue
                
            earliest_date = hist.index[0].tz_localize(None)
            latest_date = hist.index[-1].tz_localize(None)
            latest_price = hist['Close'].iloc[-1]
            
            # Check if it has traded recently (within the last 10 days to account for long weekends/holidays)
            is_actively_trading = (now - latest_date).days <= 10
            
            # Check if it covers our required window AND is above $5 AND is still actively trading
            if earliest_date.year <= start_year and latest_price >= min_price and is_actively_trading:
                valid_tickers.append(ticker)
                
        except Exception as e:
            # Silently skip errors (like completely delisted tickers yfinance can't even query)
            continue
            
    print(f"Kept {len(valid_tickers)} valid, actively trading tickers.")
    return valid_tickers

# ==========================================
# 3. MAIN BUILD FUNCTION
# ==========================================
def build_and_scale_dataset():
    # 1. Load from file and Filter the Finviz list
    raw_tickers = load_tickers_from_file(TICKERS_FILE)
    valid_energy_stocks = filter_valid_stocks(raw_tickers)
    
    if not valid_energy_stocks:
        raise ValueError("No valid stocks remained after filtering. Check your text file and constraints.")
    
    # 2. Download XLE (Target Setup)
    print(f"\nDownloading Target: {TARGET_TICKER}")
    df = yf.download(TARGET_TICKER, start=START_DATE, end=END_DATE)[['Close']]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.rename(columns={'Close': 'XLE_Close'}, inplace=True)
    
    # Calculate daily LOG return for Target: ln(today / yesterday)
    # Using .clip(lower=0.01) as a safety net to prevent np.log(0) or negative numbers
    safe_xle_close = df['XLE_Close'].clip(lower=0.01)
    df['XLE_Log_Return'] = np.log(safe_xle_close / safe_xle_close.shift(1))
    
    # Create Binary Target: 1 for Green (Return > 0), 0 for Red (Return <= 0)
    df['XLE_Green_Today'] = (df['XLE_Log_Return'] > 0).astype(int)
    
    # SHIFT TARGET TO TOMORROW (To prevent data leakage)
    df['Target_Next_Day_Direction'] = df['XLE_Green_Today'].shift(-1)
    
    # Drop intermediate target calculation columns to avoid model cheating
    df.drop(columns=['XLE_Close', 'XLE_Log_Return', 'XLE_Green_Today'], inplace=True)

    # 3. Download Macro Features
    print("\nDownloading Macro Features...")
    for ticker, name in MACRO_TICKERS.items():
        feat_data = yf.download(ticker, start=START_DATE, end=END_DATE)[['Close']]
        if isinstance(feat_data.columns, pd.MultiIndex):
            feat_data.columns = feat_data.columns.droplevel(1)
            
        close_series = feat_data['Close']
        
        # Calculate log return for ALL macro features
        safe_macro_close = close_series.clip(lower=0.01)
        df[f'{name}_Log_Return'] = np.log(safe_macro_close / safe_macro_close.shift(1))

    # 4. Download Individual Energy Stock Features
    print("\nDownloading Individual Energy Stocks...")
    stock_data = yf.download(valid_energy_stocks, start=START_DATE, end=END_DATE)['Close']
    
    # Calculate daily log returns for all individual stocks
    safe_stock_data = stock_data.clip(lower=0.01)
    stock_log_returns = np.log(safe_stock_data / safe_stock_data.shift(1))
    
    # Rename columns to reflect log returns
    stock_log_returns.columns = [f"{col}_Log_Return" for col in stock_log_returns.columns]
    
    # Merge with main dataframe
    df = df.join(stock_log_returns)

    # 5. Data Cleaning
    print("\nCleaning Data...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Catch any infinity values generated by log limits
    df.dropna(how='all', inplace=True) 
    df.ffill(inplace=True)             
    df.bfill(inplace=True)             
    
    df.dropna(subset=['Target_Next_Day_Direction'], inplace=True)
    
    # 6. Time Period Splitting
    df.index = df.index.tz_localize(None)
    
    train_df = df.loc['2016-01-01':'2023-12-31']
    val_df   = df.loc['2024-01-01':'2024-12-31']
    test_df  = df.loc['2025-01-01':'2025-12-31']
    
    # 7. Feature Scaling 
    print("\nScaling Features...")
    
    X_train = train_df.drop(columns=['Target_Next_Day_Direction'])
    y_train = train_df['Target_Next_Day_Direction']
    
    X_val = val_df.drop(columns=['Target_Next_Day_Direction'])
    y_val = val_df['Target_Next_Day_Direction']
    
    X_test = test_df.drop(columns=['Target_Next_Day_Direction'])
    y_test = test_df['Target_Next_Day_Direction']

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = build_and_scale_dataset()
    
    print("\n--- Dataset Pipeline Complete ---")
    print(f"Training set (2018-2023):   {X_train.shape[0]} days, {X_train.shape[1]} features")
    print(f"Validation set (2024):      {X_val.shape[0]} days, {X_val.shape[1]} features")
    print(f"Testing set (2025):         {X_test.shape[0]} days, {X_test.shape[1]} features")
    
    print("\nTraining Target Distribution:")
    print(y_train.value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    
    X_train.to_csv('X_train(2016-2023).csv')
    y_train.to_csv('y_train1(2016-2023).csv')
    X_val.to_csv('X_val(2024).csv')
    y_val.to_csv('y_val(2024).csv')
    X_test.to_csv('X_test(2025).csv')
    y_test.to_csv('y_test(2025).csv')
