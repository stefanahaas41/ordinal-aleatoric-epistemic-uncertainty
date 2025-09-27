import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    obesity_df = pd.read_csv(
        './data/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition/ObesityDataSet_raw_and_data_sinthetic.csv',
        sep=',')

    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Insufficient_Weight", 0)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Normal_Weight", 1)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Overweight_Level_I", 2)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Overweight_Level_II", 3)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Obesity_Type_I", 4)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Obesity_Type_II", 5)
    obesity_df['NObeyesdad'] = obesity_df['NObeyesdad'].replace("Obesity_Type_III", 6)

    # Encode labels
    le = LabelEncoder()
    le.fit(obesity_df['NObeyesdad'])

    enc = ColumnTransformer(
        [("one_hot_enc", OneHotEncoder(), [0, 4, 5, 8, 9, 11, 14, 15])], remainder='passthrough')

    return experiment(obesity_df, enc, 'NObeyesdad', le, "Obesity")


if __name__ == '__main__':
    run()
