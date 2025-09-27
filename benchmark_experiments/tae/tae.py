import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    tae_df = pd.read_csv('./data/teaching+assistant+evaluation/tae.data', sep=',',
                         names=['english', 'instr', 'course', 'term', 'class_size', 'label'])

    # Encode labels
    le = LabelEncoder()
    le.fit(tae_df['label'])

    return experiment(tae_df, None, 'label', le, "TAE")


if __name__ == '__main__':
    run()
