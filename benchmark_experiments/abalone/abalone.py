import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    abalone_df = pd.read_csv('./data/abalone.ord',sep=" ", header=None)
    abalone_df.columns = [f'Feature {i}' if i < (len(abalone_df.columns) - 1) else "label" for i in range(len(abalone_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(abalone_df["label"])


    return experiment(abalone_df, None, 'label', le, "Abalone")


if __name__ == '__main__':
    run()
