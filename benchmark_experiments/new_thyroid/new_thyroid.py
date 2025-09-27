import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    new_thyroid_df = pd.read_csv('./data/thyroid+disease/new-thyroid.data', sep=',',
                              names=["Class","T3-resin","Total Serum thyroxin","Total serum triiodothyronine",
                                      "basal thyroid-stimulating hormone (TSH)","Maximal absolute difference of TSH"])
    # Encode labels
    le = LabelEncoder()
    le.fit(new_thyroid_df['Class'])

    enc = None

    return experiment(new_thyroid_df, enc, 'Class', le, "New Thyroid")


if __name__ == '__main__':
    run()
