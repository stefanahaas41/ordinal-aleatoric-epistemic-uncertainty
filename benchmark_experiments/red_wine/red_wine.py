import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    red_wine_df = pd.read_csv('./data/wine+quality/winequality-red.csv', sep=';')

    # Encode labels
    le = LabelEncoder()
    le.fit(red_wine_df['quality'])

    enc = None

    return experiment(red_wine_df, enc, 'quality', le, "Red Wine")


if __name__ == '__main__':
    run()
