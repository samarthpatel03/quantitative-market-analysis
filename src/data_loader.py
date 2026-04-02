import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    """Download OHLCV data for a given ticker and date range."""
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.droplevel(1)
    return df