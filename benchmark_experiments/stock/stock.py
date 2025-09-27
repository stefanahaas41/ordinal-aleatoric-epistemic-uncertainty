import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    stock_df = pd.read_csv('./data/stock.ord',sep=" ", header=None)
    stock_df.columns = [f'Feature {i}' if i < (len(stock_df.columns) - 1) else "label" for i in range(len(stock_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(stock_df["label"])


    return experiment(stock_df, None, 'label', le, "Stocks Domain")


if __name__ == '__main__':
    run()
