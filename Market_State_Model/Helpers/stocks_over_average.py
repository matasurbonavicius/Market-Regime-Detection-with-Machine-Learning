import yfinance as yf
import pandas as pd
import bs4 as bs
import pickle
import requests


def historical_stocks_above_moving_averages(csv_file: str, windows: list, period: str = '1d') -> pd.DataFrame:
    
    """
    Fetches stock data from yfinance, computes historically how many of them are 
    above their respective moving averages, and returns the results as a DataFrame.
    """


    tickers_df = pd.read_csv(csv_file)
    tickers_list = tickers_df['Ticker'].tolist()
    
    data = yf.download(tickers_list, start="2000-01-01", end="2023-09-01", interval=period)

    data = data['Adj Close']

    proportion_data = {}
    for window in windows:
        mas  = data.rolling(window=window).mean()
        above = (data > mas).astype(int)
        proportion_above = above.sum(axis=1) / len(tickers_list)

        column_name = str(window)
        proportion_data[column_name] = proportion_above

    proportion_df = pd.DataFrame(proportion_data)
    return proportion_df


# Example usage
periods_to_check = [4, 12, 16, 20, 24]
result = historical_stocks_above_moving_averages("Files/sp500tickers.csv", periods_to_check)
result.to_csv('Files/stocks_over_average.csv')
print(result)


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()  # Using .strip() to remove any whitespace
        tickers.append(ticker)
    
    # Convert list of tickers into a DataFrame
    df = pd.DataFrame(tickers, columns=['Ticker'])
    
    # Save the DataFrame as a CSV
    df.to_csv("Files/sp500tickers.csv", index=False)
    
    return tickers

# save_sp500_tickers()