import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    lev_df = loadarff('./data/LEV.arff')
    lev_df = pd.DataFrame(lev_df[0])

    # Encode labels
    le = LabelEncoder()
    le.fit(lev_df['Out1'])


    enc = None

    return experiment(lev_df, enc, 'Out1', le, "LEV")


if __name__ == '__main__':
    run()
