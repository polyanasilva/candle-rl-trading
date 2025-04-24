import pandas as pd
from sklearn.preprocessing import Normalizer


class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.original_df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath, index_col=0, parse_dates=True)

    def select_columns(self):
        columns = ['Date', 'Open', 'High', 'Low', 'Close']
        self.df = self.df[columns]

    def copy_original(self):
        self.original_df = self.df.copy()

    def normalize(self):
        normalizer = Normalizer()
        self.df[['Open', 'High', 'Low', 'Close']] = normalizer.fit_transform(
            self.df[['Open', 'High', 'Low', 'Close']]
        )

    def add_candle_features(self):
        self.df['body'] = (self.df['Close'] - self.df['Open']) / self.df['Open'] * 100
        self.df['upper_shadow'] = (self.df['HIgh'] - self.df[['Open', 'Close']].max(axis=1)) / self.df['Open'] * 100
        self.df['lower_shadow'] = (self.df[['Open', 'Close']].min(axis=1) - self.df['Low']) / self.df['Open'] * 100    

    def remove_outliers(self):
        for col in ['body', 'upper_shadow', 'lower_shadow']:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

