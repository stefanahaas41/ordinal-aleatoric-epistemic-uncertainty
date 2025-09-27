import pandas as pd
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    balance_df = pd.read_csv('./data/balance+scale/balance-scale.data', names=["Class Name", "Left-Weight",
                                                                                "Left-Distance", "Right-Weight",
                                                                                "Right-Distance"])

    balance_df["Class Name"] = balance_df['Class Name'].replace('L', 1)
    balance_df['Class Name'] = balance_df['Class Name'].replace('B', 2)
    balance_df['Class Name'] = balance_df['Class Name'].replace('R', 3)


    # Encode labels
    le = LabelEncoder()
    le.fit(balance_df["Class Name"])

    enc = None

    return experiment(balance_df, enc, "Class Name", le, "Balance Scale")


if __name__ == '__main__':
    run()
