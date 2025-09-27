import pandas as pd
from category_encoders import OneHotEncoder
from scipy.io.arff import loadarff
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from benchmark_experiments.script import experiment


def run():
    eucalyptus_df = loadarff('./data/dataset_194_eucalyptus.arff',
                             )
    eucalyptus_df = pd.DataFrame(eucalyptus_df[0],
                                 columns=["Abbrev", "Rep", "Locality", "Map_Ref", "Latitude", "Altitude", "Rainfall",
                                          "Frosts", "Year",
                                          "Sp", "PMCno", "DBH", "Ht", "Surv", 'Vig', 'Ins_res', 'Stem_Fm', 'Crown_Fm',
                                          'Brnch_Fm', 'Utility'])

    eucalyptus_df['Utility'] = eucalyptus_df['Utility'].replace(b'none', 0)
    eucalyptus_df['Utility'] = eucalyptus_df['Utility'].replace(b'low', 1)
    eucalyptus_df['Utility'] = eucalyptus_df['Utility'].replace(b'average', 2)
    eucalyptus_df['Utility'] = eucalyptus_df['Utility'].replace(b'good', 3)
    eucalyptus_df['Utility'] = eucalyptus_df['Utility'].replace(b'best', 4)

    eucalyptus_df[['PMCno', "DBH", "Ht", "Surv", 'Vig', 'Ins_res', 'Stem_Fm', 'Crown_Fm',
                   'Brnch_Fm']] = eucalyptus_df[['PMCno', "DBH", "Ht", "Surv", 'Vig', 'Ins_res', 'Stem_Fm', 'Crown_Fm',
                                                 'Brnch_Fm']].fillna(0)

    # Encode labels
    le = LabelEncoder()
    le.fit(eucalyptus_df['Utility'])

    ct = ColumnTransformer(
        [("norm1", OneHotEncoder(), [0, 2, 3, 4, 9])], remainder='passthrough')

    return experiment(eucalyptus_df, ct, 'Utility', le, "Eucalyptus")


if __name__ == '__main__':
    run()
