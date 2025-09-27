import pandas as pd
from category_encoders import OneHotEncoder
from scipy.io.arff import loadarff
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    grub_damage_arff = loadarff("./data/phpnYQXoc.arff")

    grub_damage_df = pd.DataFrame(grub_damage_arff[0])

    grub_damage_df['year'] = grub_damage_df['year'].astype(int)
    grub_damage_df['damage_rankRJT'] = grub_damage_df['damage_rankRJT'].astype(int)
    grub_damage_df['damage_rankALL'] = grub_damage_df['damage_rankALL'].astype(int)

    grub_damage_df['year_zone'] = grub_damage_df['year_zone'].astype(str)
    grub_damage_df['dry_or_irr'] = grub_damage_df['dry_or_irr'].astype(str)
    grub_damage_df['zone'] = grub_damage_df['zone'].astype(str)

    grub_damage_df['GG_new'] = grub_damage_df['GG_new'].replace(b'low', 0)
    grub_damage_df['GG_new'] = grub_damage_df['GG_new'].replace(b'average', 1)
    grub_damage_df['GG_new'] = grub_damage_df['GG_new'].replace(b'high', 2)
    grub_damage_df['GG_new'] = grub_damage_df['GG_new'].replace(b'veryhigh', 3)

    # Encode labels
    le = LabelEncoder()
    le.fit(grub_damage_df['GG_new'])

    # Encode categorical data
    enc = ColumnTransformer(
        [("one_hot_enc", OneHotEncoder(), [0, 6, 7])], remainder='passthrough')

    return experiment(grub_damage_df, enc, 'GG_new', le, "Grub Damage")


if __name__ == '__main__':
    run()
