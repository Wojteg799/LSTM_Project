import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, ticker='NVDA', start_date='2020-01-01', end_date='2026-01-01', lookback=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None

    def fetch_data(self):
        """Fetches historical stock data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        try:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            # Ensure we only use 'Close' price and drop any NaNs immediately
            if 'Close' not in self.data.columns:
                 # Handle cases where columns might be MultiIndex (e.g. Price, Ticker)
                 # yfinance structure can vary based on version
                 if isinstance(self.data.columns, pd.MultiIndex):
                     try:
                         self.data = self.data.xs('Close', level='Price', axis=1)
                     except KeyError:
                         # Fallback if structure is different
                         print("Warning: Could not find 'Close' in MultiIndex, trying default column access.")
            
            if 'Close' in self.data:
                 self.data = self.data[['Close']]
            
            self.data = self.data.dropna()
            print(f"Data fetched successfully. {len(self.data)} rows.")
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, split_ratio=0.8):
        """Normalizes data and splits into train/test sets."""
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")

        dataset = self.data.values
        training_data_len = int(len(dataset) * split_ratio)

        self.scaled_data = self.scaler.fit_transform(dataset)

        train_data = self.scaled_data[0:training_data_len, :]
        test_data = self.scaled_data[training_data_len - self.lookback:, :]

        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)

        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, self.scaler

    def create_sequences(self, data):
        """Creates sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def get_original_data(self):
        return self.data
