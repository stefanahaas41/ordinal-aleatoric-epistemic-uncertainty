import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    white_wine_df = pd.read_csv('./data/wine+quality/winequality-white.csv', sep=';')

    # Encode labels
    le = LabelEncoder()
    le.fit(white_wine_df['quality'])

    enc = None

    return experiment(white_wine_df, enc, 'quality', le, "White Wine")


if __name__ == '__main__':
    run()
