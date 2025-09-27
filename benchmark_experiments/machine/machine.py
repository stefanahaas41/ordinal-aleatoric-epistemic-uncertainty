import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    machine_df = pd.read_csv('./data/machine.ord',sep=" ", header=None)
    machine_df.columns = [f'Feature {i}' if i < (len(machine_df.columns) - 1) else "label" for i in range(len(machine_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(machine_df["label"])


    return experiment(machine_df, None, 'label', le, "Machine CPU")


if __name__ == '__main__':
    run()
