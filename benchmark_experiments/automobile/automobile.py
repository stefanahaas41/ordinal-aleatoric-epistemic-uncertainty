import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    automobile_df = pd.read_csv('./data/automobile/imports-85.data',
                                na_values='?', sep=",",
                                names=["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
                                       "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base",
                                       "length", "width", "height", "curb-weight", "engine-type",
                                       "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
                                       "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
                                       "price"])

    automobile_df.fillna(0, inplace=True)

    # Encode labels
    le = LabelEncoder()
    le.fit(automobile_df['symboling'])

    ct = ColumnTransformer(
        [("norm1", OneHotEncoder(), [1, 2, 3, 4, 5, 6, 7, 13, 14, 16])], remainder='passthrough')

    return experiment(automobile_df, ct, 'symboling', le, "Automobile")


if __name__ == '__main__':
    run()
