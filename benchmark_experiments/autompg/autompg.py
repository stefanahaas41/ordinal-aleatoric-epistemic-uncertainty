import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    autompg_df = pd.read_csv('./data/auto.data.ord',sep=" ", header=None)
    autompg_df.columns = [f'Feature {i}' if i < (len(autompg_df.columns) - 1) else "label" for i in range(len(autompg_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(autompg_df["label"])


    return experiment(autompg_df, None, 'label', le, "Auto MPG")


if __name__ == '__main__':
    run()
