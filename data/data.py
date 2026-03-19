from finvizfinance.screener.overview import Overview
import pandas as pd


def get_energy_stocks():
    foverview = Overview()

    filters_dict = {
        'Sector': 'Energy',
        'Price': 'Over $5',
        'Exchange': 'NASDAQ'
    }

    try:
        foverview.set_filter(filters_dict=filters_dict)
        df = foverview.screener_view()

        if df.empty:
            print("No stocks found")
            return []

        tickers = df['Ticker'].tolist()

        with open("energy_tickers.txt", "w") as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")

        print(
            f"Success! Found {len(tickers)} stocks and saved to 'energy_tickers_finviz.txt'.")
        return tickers

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


energy_tickers = get_energy_stocks()
# verify against: https://stockanalysis.com/stocks/screener/
