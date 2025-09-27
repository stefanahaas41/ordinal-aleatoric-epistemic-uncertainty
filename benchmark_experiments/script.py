import os
import random
from enum import Enum

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from dlordinal.losses import WKLoss, BetaCrossEntropyLoss, TriangularCrossEntropyLoss
from lightgbm import LGBMClassifier
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch import nn
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from xgboost import XGBClassifier

from custom_estimators.LGBMUnimodalSoftLabelClassifier import LGBMUnimodalSoftLabelClassifier
from custom_estimators.RPSLoss import RPSLoss
from util.measures.binary_decompositions import binary_ordinal_variance, eu_binary_variance, binary_ordinal_entropy, \
    binary_one_vs_rest_entropy, binary_one_vs_rest_variance, eu_binary_variance_one_vs_rest
from util.measures.variance import variance_for_probabilities, mean_for_probabilities
from util.uncertainty.rejection_curve import rejection_curve

torch.set_default_dtype(torch.float32)


# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustMLPModule(nn.Module):
    def __init__(
            self,
            input_units=20,
            output_units=2,
            apply_softmax=False
    ):
        super().__init__()

        hidden_units = [128, 64, 32]

        # Define the layers
        in_layer = nn.Linear(in_features=input_units, out_features=hidden_units[0], bias=True)
        hidden_layer1 = nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1], bias=True)
        hidden_layer2 = nn.Linear(in_features=hidden_units[1], out_features=hidden_units[2], bias=True)
        out_layer2 = nn.Linear(in_features=hidden_units[2], out_features=output_units, bias=True)

        # Define the sequence of layers
        layers = [
            in_layer,
            nn.ReLU(),
            hidden_layer1,
            nn.ReLU(),
            hidden_layer2,
            nn.ReLU(),
            out_layer2
        ]

        # Optionally apply Softmax
        if apply_softmax:
            layers.append(Softmax(dim=-1))

        self.sequential = nn.Sequential(
            *layers
        )

    def forward(self, X):
        X = self.sequential(X)
        return X


class ExperimentType(Enum):
    ERROR = 1
    OOD = 2


experiment_type = ExperimentType.ERROR


def prepare_mlp(X_test, X_train, y_train):
    num_features = X_train.shape[1]
    num_classes = len(set(y_train))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_test, X_train, num_classes, num_features


clfs = {
    'ENS (MLP) QWK': VotingClassifier(estimators=[], voting='soft'),
    'ENS (MLP) EMD': VotingClassifier(estimators=[], voting='soft'),
    'ENS (MLP) TRI': VotingClassifier(estimators=[], voting='soft'),
    'ENS (MLP) BETA': VotingClassifier(estimators=[], voting='soft'),
    'ENS (LGBM-GEO)': VotingClassifier(estimators=[
        ('lgb1',
         LGBMUnimodalSoftLabelClassifier(alpha=0.1)),
        ('lgb2',
         LGBMUnimodalSoftLabelClassifier(alpha=0.2)),
        ('lgb3',
         LGBMUnimodalSoftLabelClassifier(alpha=0.3)),
        ('lgb4',
         LGBMUnimodalSoftLabelClassifier(alpha=0.05)),
        ('lgb5',
         LGBMUnimodalSoftLabelClassifier(alpha=0.15)),
        ('lgb6',
         LGBMUnimodalSoftLabelClassifier(alpha=0.25)),
        ('lgb7',
         LGBMUnimodalSoftLabelClassifier(alpha=0.35)),
        ('lgb8',
         LGBMUnimodalSoftLabelClassifier(alpha=0.4)),
        ('lgb9',
         LGBMUnimodalSoftLabelClassifier(alpha=0.45)),
        ('lgb10',
         LGBMUnimodalSoftLabelClassifier(alpha=0.5))
    ], voting='soft'),
    'ENS (LGBM)': VotingClassifier(estimators=[
        ('lgb1', LGBMClassifier(random_state=1, subsample=0.5, subsample_freq=1)),
        ('lgb2', LGBMClassifier(random_state=2, subsample=0.5, subsample_freq=1)),
        ('lgb3', LGBMClassifier(random_state=3, subsample=0.5, subsample_freq=1)),
        ('lgb4', LGBMClassifier(random_state=4, subsample=0.5, subsample_freq=1)),
        ('lgb5', LGBMClassifier(random_state=5, subsample=0.5, subsample_freq=1)),
        ('lgb6', LGBMClassifier(random_state=6, subsample=0.5, subsample_freq=1)),
        ('lgb7', LGBMClassifier(random_state=7, subsample=0.5, subsample_freq=1)),
        ('lgb8', LGBMClassifier(random_state=8, subsample=0.5, subsample_freq=1)),
        ('lgb9', LGBMClassifier(random_state=9, subsample=0.5, subsample_freq=1)),
        ('lgb10', LGBMClassifier(random_state=10, subsample=0.5, subsample_freq=1))
    ], voting='soft'),
    'ENS (MLP)': VotingClassifier(estimators=[
        ('mlp1', MLPClassifier(random_state=1, hidden_layer_sizes=(128, 64, 32))),
        ('mlp2', MLPClassifier(random_state=2, hidden_layer_sizes=(128, 64, 32))),
        ('mlp3', MLPClassifier(random_state=3, hidden_layer_sizes=(128, 64, 32))),
        ('mlp4', MLPClassifier(random_state=4, hidden_layer_sizes=(128, 64, 32))),
        ('mlp5', MLPClassifier(random_state=5, hidden_layer_sizes=(128, 64, 32))),
        ('mlp6', MLPClassifier(random_state=6, hidden_layer_sizes=(128, 64, 32))),
        ('mlp7', MLPClassifier(random_state=7, hidden_layer_sizes=(128, 64, 32))),
        ('mlp8', MLPClassifier(random_state=8, hidden_layer_sizes=(128, 64, 32))),
        ('mlp9', MLPClassifier(random_state=9, hidden_layer_sizes=(128, 64, 32))),
        ('mlp10', MLPClassifier(random_state=10, hidden_layer_sizes=(128, 64, 32)))
    ], voting='soft'),
    'ENS (CAT)': VotingClassifier(estimators=[
        ('cat1', CatBoostClassifier(random_state=1, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat2', CatBoostClassifier(random_state=2, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat3', CatBoostClassifier(random_state=3, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat4', CatBoostClassifier(random_state=4, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat5', CatBoostClassifier(random_state=5, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat6', CatBoostClassifier(random_state=6, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat7', CatBoostClassifier(random_state=7, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat8', CatBoostClassifier(random_state=8, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat9', CatBoostClassifier(random_state=9, subsample=0.5, bootstrap_type='Bernoulli')),
        ('cat10', CatBoostClassifier(random_state=10, subsample=0.5, bootstrap_type='Bernoulli'))
    ], voting='soft'),
    'ENS (XGB)': VotingClassifier(estimators=[
        ('xgb1', XGBClassifier(random_state=1, subsample=0.5)),
        ('xgb2', XGBClassifier(random_state=2, subsample=0.5)),
        ('xgb3', XGBClassifier(random_state=3, subsample=0.5)),
        ('xgb4', XGBClassifier(random_state=4, subsample=0.5)),
        ('xgb5', XGBClassifier(random_state=5, subsample=0.5)),
        ('xgb6', XGBClassifier(random_state=6, subsample=0.5)),
        ('xgb7', XGBClassifier(random_state=7, subsample=0.5)),
        ('xgb8', XGBClassifier(random_state=8, subsample=0.5)),
        ('xgb9', XGBClassifier(random_state=9, subsample=0.5)),
        ('xgb10', XGBClassifier(random_state=10, subsample=0.5))
    ], voting='soft')
}


def experiment(data_df, enc, label, le, dataset):
    if experiment_type == ExperimentType.OOD:
        df_ood = pd.read_csv('./data/YearPredictionMSD.txt', header=None).dropna()

    for clf_name, clf in clfs.items():

        print(f"Clf {clf_name}:")

        result = []

        overall_result_rejection_curve = {
            "Rejection": [],
            "Performance": [],
            "Metric": [],
            "Measure": []
        }

        # set_config(transform_output="pandas")

        X = data_df.copy()
        y = X.pop(label)

        print(f"Dataset {dataset}:")

        predicted_proba = []
        true_class = []

        kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Fold {i}:")

            X_train = data_df.iloc[train_index]
            X_test = data_df.iloc[test_index]

            y_train = X_train.pop(label)
            y_test = X_test.pop(label)

            X_test_raw = X_test.copy()

            y_train = le.transform(y_train)

            num_classes = len(np.unique(y_train))

            if enc:
                # For MLP we also need to standardize numerical features
                if 'ENS (MLP)' in clf_name:
                    pipe = Pipeline(steps=[
                        ('encoder', enc),
                        ('scaler', StandardScaler())
                    ])
                    X_train = pipe.fit_transform(X_train)
                    X_test = pipe.transform(X_test)
                else:
                    X_train = enc.fit_transform(X_train)
                    X_test = enc.transform(X_test)
            else:
                # For MLP we need to standardize numerical features
                if 'ENS (MLP)' in clf_name:
                    pipe = Pipeline(steps=[('scaler', StandardScaler())])
                    X_train = pipe.fit_transform(X_train)
                    X_test = pipe.transform(X_test)

            if clf_name == 'ENS (MLP) EMD':
                X_test, X_train = add_estimators(X_test, X_train, clf, y_train, apply_softmax=True, loss=RPSLoss())
            if clf_name == 'ENS (MLP) QWK':
                X_test, X_train = add_estimators(X_test, X_train, clf, y_train, apply_softmax=True,
                                                 loss=WKLoss(num_classes=num_classes))
            if clf_name == 'ENS (MLP) TRI':
                X_test, X_train = add_estimators(X_test, X_train, clf, y_train, apply_softmax=True,
                                                 loss=TriangularCrossEntropyLoss(num_classes=num_classes))
            if clf_name == 'ENS (MLP) BETA':
                X_test, X_train = add_estimators(X_test, X_train, clf, y_train, apply_softmax=True,
                                                 loss=BetaCrossEntropyLoss(num_classes=num_classes))

            clf.fit(X_train, y_train)

            # Predict
            estimator_results_stacked = get_single_estimator_predictions(X_test, clf, clf_name)
            y_pred_proba = clf.predict_proba(X_test)
            if clf_name == 'ENS (LGBM-GEO)':
                y_pred_proba = softmax(y_pred_proba, axis=1)

            # Save prediction and true result
            predicted_proba.extend(y_pred_proba)
            true_class.extend(y_test)

            # Get uncertainties
            (au_bin_ent, au_bin_one_vs_rest_ent, au_bin_one_vs_rest_var, au_bin_var,
             au_entropy, au_variance, eu_bin_ent, eu_bin_one_vs_rest_ent, eu_bin_one_vs_rest_var,
             eu_bin_var, eu_entropy, eu_variance, tu_bin_ent, tu_bin_one_vs_rest_ent, tu_bin_one_vs_rest_var,
             tu_bin_var, tu_entropy, tu_variance) = get_uncertainties(
                estimator_results_stacked, y_pred_proba)

            if experiment_type == ExperimentType.ERROR:
                # Get PRR values and rejection curves
                rejection_curves, prr, _ = rejection_curve(['ACC', 'MAE', 'MSE'], {
                    'Random': random,
                    'Entropy (TU)': tu_entropy,
                    'Entropy (EU)': eu_entropy,
                    'Entropy (AU)': au_entropy,
                    'Binary Variance (TU)': tu_bin_var,
                    'Binary Variance (EU)': eu_bin_var,
                    'Binary Variance (AU)': au_bin_var,
                    'Binary Entropy (TU)': tu_bin_ent,
                    'Binary Entropy (EU)': eu_bin_ent,
                    'Binary Entropy (AU)': au_bin_ent,
                    'Variance (TU)': tu_variance,
                    'Variance (EU)': eu_variance,
                    'Variance (AU)': au_variance,
                    'Binary Entropy One vs. Rest (TU)': tu_bin_one_vs_rest_ent,
                    'Binary Entropy One vs. Rest (EU)': eu_bin_one_vs_rest_ent,
                    'Binary Entropy One vs. Rest (AU)': au_bin_one_vs_rest_ent,
                    'Binary Variance One vs. Rest (TU)': tu_bin_one_vs_rest_var,
                    'Binary Variance One vs. Rest (EU)': eu_bin_one_vs_rest_var,
                    'Binary Variance One vs. Rest (AU)': au_bin_one_vs_rest_var

                }, y_pred_proba, y_test, le)

                overall_result_rejection_curve["Rejection"].extend(rejection_curves["Rejection"])
                overall_result_rejection_curve["Performance"].extend(rejection_curves["Performance"])
                overall_result_rejection_curve["Measure"].extend(rejection_curves["Measure"])
                overall_result_rejection_curve["Metric"].extend(rejection_curves["Metric"])

                result.append(['Entropy', 'TU', dataset, prr['ACC'][0], prr['MAE'][0], prr['MSE'][0]])
                result.append(['Entropy', 'EU', dataset, prr['ACC'][1], prr['MAE'][1], prr['MSE'][1]])
                result.append(['Entropy', 'AU', dataset, prr['ACC'][2], prr['MAE'][2], prr['MSE'][2]])
                result.append(['Binary Variance', 'TU', dataset, prr['ACC'][3], prr['MAE'][3], prr['MSE'][3]])
                result.append(['Binary Variance', 'EU', dataset, prr['ACC'][4], prr['MAE'][4], prr['MSE'][4]])
                result.append(['Binary Variance', 'AU', dataset, prr['ACC'][5], prr['MAE'][5], prr['MSE'][5]])
                result.append(['Binary Entropy', 'TU', dataset, prr['ACC'][6], prr['MAE'][6], prr['MSE'][6]])
                result.append(['Binary Entropy', 'EU', dataset, prr['ACC'][7], prr['MAE'][7], prr['MSE'][7]])
                result.append(['Binary Entropy', 'AU', dataset, prr['ACC'][8], prr['MAE'][8], prr['MSE'][8]])
                result.append(['Variance', 'TU', dataset, prr['ACC'][9], prr['MAE'][9], prr['MSE'][9]])
                result.append(['Variance', 'EU', dataset, prr['ACC'][10], prr['MAE'][10], prr['MSE'][10]])
                result.append(['Variance', 'AU', dataset, prr['ACC'][11], prr['MAE'][11], prr['MSE'][11]])
                result.append(
                    ['Binary Variance One vs. Rest', 'TU', dataset, prr['ACC'][12], prr['MAE'][12], prr['MSE'][12]])
                result.append(
                    ['Binary Variance One vs. Rest', 'EU', dataset, prr['ACC'][13], prr['MAE'][13], prr['MSE'][13]])
                result.append(
                    ['Binary Variance One vs. Rest', 'AU', dataset, prr['ACC'][14], prr['MAE'][14], prr['MSE'][14]])
                result.append(
                    ['Binary Entropy One vs. Rest', 'TU', dataset, prr['ACC'][15], prr['MAE'][15], prr['MSE'][15]])
                result.append(
                    ['Binary Entropy One vs. Rest', 'EU', dataset, prr['ACC'][16], prr['MAE'][16], prr['MSE'][16]])
                result.append(
                    ['Binary Entropy One vs. Rest', 'AU', dataset, prr['ACC'][17], prr['MAE'][17], prr['MSE'][17]])

            elif experiment_type == ExperimentType.OOD or experiment_type == ExperimentType.OOD_ORD:

                if experiment_type == ExperimentType.OOD:
                    # Sample dataset of the same size from Year MSD dataset
                    X_ood_test = X_test_raw.copy()
                    i = 0
                    for column in X_ood_test.columns:
                        # For categorical sample a random category uniformly at random from all of the feature's categories
                        if isinstance(X_ood_test[column], pd.CategoricalDtype) or \
                                X_ood_test[column].dtype == 'O' or \
                                X_ood_test[column].dtype == 'string' or \
                                pd.api.types.is_datetime64_any_dtype(X_ood_test[column]) or \
                                pd.api.types.is_timedelta64_dtype(X_ood_test[column]) or \
                                X_ood_test[column].dtype == 'bool':
                            # Get the unique categories
                            categories = X_ood_test[column].unique()
                            # Randomly sample a category
                            X_ood_test[column] = categories[
                                np.random.randint(0, len(categories), size=len(X_ood_test[column]))]
                        else:
                            # All numerical features are normalized by the per-column mean and variance from the IDD data
                            mean = np.mean(X_ood_test[column])
                            variance = np.var(X_ood_test[column])

                            df_ood_sample = df_ood.sample(n=X_test.shape[0], random_state=42).sample(n=1, axis=1,
                                                                                                     random_state=42)
                            # Handle columns where variance is 0 (set to 0)
                            if variance == 0:
                                X_ood_test[column] = 0
                            else:
                                X_ood_test[column] = ((df_ood_sample.iloc[:, 0] - mean) / variance).to_numpy()

                        i += 1

                # Preprocess OOD dataset
                if enc:
                    # For MLP we also need to standardize numerical features
                    if 'ENS (MLP)' in clf_name:
                        X_ood_test = pipe.transform(X_ood_test)
                    else:
                        X_ood_test = enc.transform(X_ood_test)

                else:
                    # For MLP we need to standardize numerical features
                    if 'ENS (MLP)' in clf_name:
                        X_ood_test = pipe.transform(X_ood_test)

                # Predict
                estimator_results_stacked_ood = get_single_estimator_predictions(X_ood_test, clf, clf_name)
                y_pred_proba_ood = clf.predict_proba(X_ood_test)

                # Get uncertainties
                (au_bin_ent_ood, au_bin_one_vs_rest_ent_ood, au_bin_one_vs_rest_var_ood, au_bin_var_ood,
                 au_entropy_ood, au_variance_ood, eu_bin_ent_ood, eu_bin_one_vs_rest_ent_ood,
                 eu_bin_one_vs_rest_var_ood,
                 eu_bin_var_ood, eu_entropy_ood, eu_variance_ood, tu_bin_ent_ood, tu_bin_one_vs_rest_ent_ood,
                 tu_bin_one_vs_rest_var_ood,
                 tu_bin_var_ood, tu_entropy_ood, tu_variance_ood) = get_uncertainties(
                    estimator_results_stacked_ood, y_pred_proba_ood)

                # Epistemic IDD vs. epistemic ODD
                labels = np.array([0] * len(X_test) + [1] * len(X_ood_test))

                scaler = MinMaxScaler()

                # Ord ENT
                scores_au_bin_ent = scaler.fit_transform(
                    np.concatenate([au_bin_ent, au_bin_ent_ood]).reshape(-1, 1))
                auc_roc_au_bin_ent = roc_auc_score(labels, scores_au_bin_ent)

                scores_tu_bin_ent = scaler.fit_transform(
                    np.concatenate([tu_bin_ent, tu_bin_ent_ood]).reshape(-1, 1))
                auc_roc_tu_bin_ent = roc_auc_score(labels, scores_tu_bin_ent)

                scores_eu_bin_ent = scaler.fit_transform(
                    np.concatenate([eu_bin_ent, eu_bin_ent_ood]).reshape(-1, 1))
                auc_roc_eu_bin_ent = roc_auc_score(labels, scores_eu_bin_ent)

                # Ord VAR
                scores_au_bin_var = scaler.fit_transform(
                    np.concatenate([au_bin_var, au_bin_var_ood]).reshape(-1, 1))
                auc_roc_au_bin_var = roc_auc_score(labels, scores_au_bin_var)

                scores_tu_bin_var = scaler.fit_transform(
                    np.concatenate([tu_bin_var, tu_bin_var_ood]).reshape(-1, 1))
                auc_roc_tu_bin_var = roc_auc_score(labels, scores_tu_bin_var)

                scores_eu_bin_var = scaler.fit_transform(
                    np.concatenate([eu_bin_var, eu_bin_var_ood]).reshape(-1, 1))
                auc_roc_eu_bin_var = roc_auc_score(labels, scores_eu_bin_var)

                # Bin VAR
                scores_au_bin_one_vs_rest_var = scaler.fit_transform(
                    np.concatenate([au_bin_one_vs_rest_var, au_bin_one_vs_rest_var_ood]).reshape(-1, 1))
                auc_roc_au_bin_one_vs_rest_var = roc_auc_score(labels, scores_au_bin_one_vs_rest_var)

                scores_tu_bin_one_vs_rest_var = scaler.fit_transform(
                    np.concatenate([tu_bin_one_vs_rest_var, tu_bin_one_vs_rest_var_ood]).reshape(-1, 1))
                auc_roc_tu_bin_one_vs_rest_var = roc_auc_score(labels, scores_tu_bin_one_vs_rest_var)

                scores_eu_bin_one_vs_rest_var = scaler.fit_transform(
                    np.concatenate([eu_bin_one_vs_rest_var, eu_bin_one_vs_rest_var_ood]).reshape(-1, 1))
                auc_roc_eu_bin_one_vs_rest_var = roc_auc_score(labels, scores_eu_bin_one_vs_rest_var)

                # Bin ENT
                scores_au_bin_one_vs_rest_ent = scaler.fit_transform(
                    np.concatenate([au_bin_one_vs_rest_ent, au_bin_one_vs_rest_ent_ood]).reshape(-1, 1))
                auc_roc_au_bin_one_vs_rest_ent = roc_auc_score(labels, scores_au_bin_one_vs_rest_ent)

                scores_tu_bin_one_vs_rest_ent = scaler.fit_transform(
                    np.concatenate([tu_bin_one_vs_rest_ent, tu_bin_one_vs_rest_ent_ood]).reshape(-1, 1))
                auc_roc_tu_bin_one_vs_rest_ent = roc_auc_score(labels, scores_tu_bin_one_vs_rest_ent)

                scores_eu_bin_one_vs_rest_ent = scaler.fit_transform(
                    np.concatenate([eu_bin_one_vs_rest_ent, eu_bin_one_vs_rest_ent_ood]).reshape(-1, 1))
                auc_roc_eu_bin_one_vs_rest_ent = roc_auc_score(labels, scores_eu_bin_one_vs_rest_ent)

                # ENT
                scores_au_ent = scaler.fit_transform(np.concatenate([au_entropy, au_entropy_ood]).reshape(-1, 1))
                auc_roc_au_ent = roc_auc_score(labels, scores_au_ent)

                scores_tu_ent = scaler.fit_transform(np.concatenate([tu_entropy, tu_entropy_ood]).reshape(-1, 1))
                auc_roc_tu_ent = roc_auc_score(labels, scores_tu_ent)

                scores_eu_ent = scaler.fit_transform(np.concatenate([eu_entropy, eu_entropy_ood]).reshape(-1, 1))
                auc_roc_eu_ent = roc_auc_score(labels, scores_eu_ent)

                # Variance
                scores_au_variance = scaler.fit_transform(
                    np.concatenate([au_variance, au_variance_ood]).reshape(-1, 1))
                auc_roc_au_variance = roc_auc_score(labels, scores_au_variance)

                scores_tu_variance = scaler.fit_transform(
                    np.concatenate([tu_variance, tu_variance_ood]).reshape(-1, 1))
                auc_roc_tu_variance = roc_auc_score(labels, scores_tu_variance)

                scores_eu_variance = scaler.fit_transform(
                    np.concatenate([eu_variance, eu_variance_ood]).reshape(-1, 1))
                auc_roc_eu_variance = roc_auc_score(labels, scores_eu_variance)

                result.append(['Entropy', 'TU', dataset, auc_roc_tu_ent])
                result.append(['Entropy', 'EU', dataset, auc_roc_eu_ent])
                result.append(['Entropy', 'AU', dataset, auc_roc_au_ent])

                result.append(['Binary Variance', 'TU', dataset, auc_roc_tu_bin_var])
                result.append(['Binary Variance', 'EU', dataset, auc_roc_eu_bin_var])
                result.append(['Binary Variance', 'AU', dataset, auc_roc_au_bin_var])

                result.append(['Binary Entropy', 'TU', dataset, auc_roc_tu_bin_ent])
                result.append(['Binary Entropy', 'EU', dataset, auc_roc_eu_bin_ent])
                result.append(['Binary Entropy', 'AU', dataset, auc_roc_au_bin_ent])

                result.append(['Variance', 'TU', dataset, auc_roc_tu_variance])
                result.append(['Variance', 'EU', dataset, auc_roc_eu_variance])
                result.append(['Variance', 'AU', dataset, auc_roc_au_variance])

                result.append(
                    ['Binary Variance One vs. Rest', 'TU', dataset, auc_roc_tu_bin_one_vs_rest_var])
                result.append(
                    ['Binary Variance One vs. Rest', 'EU', dataset, auc_roc_eu_bin_one_vs_rest_var])
                result.append(
                    ['Binary Variance One vs. Rest', 'AU', dataset, auc_roc_au_bin_one_vs_rest_var])

                result.append(
                    ['Binary Entropy One vs. Rest', 'TU', dataset, auc_roc_tu_bin_one_vs_rest_ent])
                result.append(
                    ['Binary Entropy One vs. Rest', 'EU', dataset, auc_roc_eu_bin_one_vs_rest_ent])
                result.append(
                    ['Binary Entropy One vs. Rest', 'AU', dataset, auc_roc_au_bin_one_vs_rest_ent])

            else:
                raise ValueError("Unknown experiment type")

        result_folder = f"./results-{clf_name}-{experiment_type.name}"
        os.makedirs(result_folder, exist_ok=True)

        if experiment_type == ExperimentType.ERROR:

            cols = ['Uncertainty Measure', 'Uncertainty Type', 'Dataset', 'PRR (MCR)', 'PRR (MAE)', 'PRR (MSE)']

            df = pd.DataFrame(result, columns=cols)
            df.to_csv(f'{result_folder}/{dataset}.csv', index=False)

            # Save rejection curves
            overall_result_df = pd.DataFrame(overall_result_rejection_curve)
            overall_result_df.to_csv(f'{result_folder}/{dataset}-rejection-curve.csv', index=False)

            # Save predictions and true classes
            df_pred = pd.DataFrame(predicted_proba)
            df_pred.to_csv(f'{result_folder}/{dataset}-pred-probas.csv', index=False)
            df_true = pd.DataFrame(true_class)
            df_true.to_csv(f'{result_folder}/{dataset}-true-classes.csv', index=False)

        elif experiment_type == ExperimentType.OOD or experiment_type == ExperimentType.OOD_ORD:

            cols = ['Uncertainty Measure', 'Uncertainty Type', 'Dataset', 'AUC-ROC']

            df = pd.DataFrame(result, columns=cols)
            df.to_csv(f'{result_folder}/{dataset}-auc-roc.csv', index=False)


def add_estimators(X_test, X_train, clf, y_train, apply_softmax=True, loss=CrossEntropyLoss()):
    clf.estimators = []
    X_test, X_train, num_classes, num_features = prepare_mlp(X_test, X_train, y_train)
    for i in range(10):
        # Set seeds for reproducibility
        set_seed(i)
        clf.estimators.append((f"ANN{i}",
                               NeuralNetClassifier(
                                   CustMLPModule,
                                   module__input_units=num_features,
                                   module__output_units=num_classes,
                                   module__apply_softmax=apply_softmax,
                                   # MLPModule(
                                   #     input_units=num_features,
                                   #     output_units=num_classes,
                                   #     hidden_units=128, num_hidden=2),
                                   max_epochs=200,
                                   batch_size=min(200, len(X_train)),
                                   optimizer=Adam,
                                   optimizer__weight_decay=0.0001,
                                   lr=0.001,
                                   train_split=ValidSplit(0.1),
                                   criterion=loss,
                                   callbacks=[
                                       EarlyStopping(patience=10, monitor="valid_loss"),
                                       LRScheduler(policy=ReduceLROnPlateau, patience=5, factor=0.5)
                                   ]
                               )))
    return X_test, X_train


def get_uncertainties(estimator_results_stacked, y_pred_proba):
    # --------------------------------------------------------
    # Ordinal Binary Variance
    # --------------------------------------------------------
    # TU (uncertainty of average)
    tu_bin_var = np.apply_along_axis(binary_ordinal_variance, 1, y_pred_proba)
    # AU (average of uncertainties)
    au_bin_var = np.apply_along_axis(binary_ordinal_variance, 1, estimator_results_stacked)
    au_bin_var = np.mean(au_bin_var, axis=1)
    # EU (variance of expectations)
    eu_bin_var = []
    for i, proba in enumerate(y_pred_proba):
        eu_bin_var.append(eu_binary_variance(estimator_results_stacked[i].T))
    eu_bin_var = np.array(eu_bin_var)
    # Assert that TU = EU + AU
    # testing.assert_allclose(tu_bin_var, au_bin_var + eu_bin_var, rtol=1e-04)
    # --------------------------------------------------------
    # Ordinal Binary Entropy
    # --------------------------------------------------------
    # TU (uncertainty of average)
    tu_bin_ent = np.apply_along_axis(binary_ordinal_entropy, 1, y_pred_proba)
    # AU (average of uncertainties)
    au_bin_ent = np.mean(np.apply_along_axis(binary_ordinal_entropy, 1, estimator_results_stacked), axis=1)
    # EU
    eu_bin_ent = tu_bin_ent - au_bin_ent
    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------
    # TU
    tu_entropy = entropy(y_pred_proba, axis=1, base=2)
    # AU
    au_entropy = np.mean(entropy(estimator_results_stacked, axis=1, base=2), axis=1)
    # EU
    eu_entropy = tu_entropy - au_entropy
    # --------------------------------------------------------
    # Variance
    # --------------------------------------------------------
    # TU
    tu_variance = np.apply_along_axis(variance_for_probabilities, 1, y_pred_proba)
    # AU
    au_variance = np.apply_along_axis(variance_for_probabilities, 1, estimator_results_stacked)
    au_variance = np.mean(au_variance, axis=1)
    # EU
    expected_values = np.apply_along_axis(mean_for_probabilities, 1, estimator_results_stacked)
    mean_expected_value = np.mean(expected_values, axis=1)
    eu_variance = np.mean((expected_values - mean_expected_value[:, np.newaxis]) ** 2, axis=1)
    # Assert that TU = EU + AU
    # testing.assert_allclose(tu_variance, au_variance + eu_variance, rtol=1e-04)
    # --------------------------------------------------------
    # Binary Entropy One vs. Rest
    # --------------------------------------------------------
    # TU (uncertainty of average)
    tu_bin_one_vs_rest_ent = np.apply_along_axis(binary_one_vs_rest_entropy, 1, y_pred_proba)
    # AU (average of uncertainties)
    au_bin_one_vs_rest_ent = np.mean(
        np.apply_along_axis(binary_one_vs_rest_entropy, 1, estimator_results_stacked), axis=1)
    # EU
    eu_bin_one_vs_rest_ent = tu_bin_one_vs_rest_ent - au_bin_one_vs_rest_ent
    # --------------------------------------------------------
    # Binary Variance One vs. Rest
    # --------------------------------------------------------
    # TU (uncertainty of average)
    tu_bin_one_vs_rest_var = np.apply_along_axis(binary_one_vs_rest_variance, 1, y_pred_proba)
    # AU (average of uncertainties)
    au_bin_one_vs_rest_var = np.mean(
        np.apply_along_axis(binary_one_vs_rest_variance, 1, estimator_results_stacked), axis=1)
    # EU
    eu_bin_one_vs_rest_var = tu_bin_one_vs_rest_var - au_bin_one_vs_rest_var
    eu_bin_var_one_vs_rest = []
    for i, proba in enumerate(y_pred_proba):
        eu_bin_var_one_vs_rest.append(eu_binary_variance_one_vs_rest(estimator_results_stacked[i].T))
    eu_bin_var_one_vs_rest = np.array(eu_bin_var_one_vs_rest)
    # Assert that TU = EU + AU
    # testing.assert_allclose(eu_bin_var_one_vs_rest, eu_bin_one_vs_rest_var, rtol=1e-03)
    # testing.assert_allclose(tu_bin_one_vs_rest_var, au_bin_one_vs_rest_var + eu_bin_var_one_vs_rest, rtol=1e-04)
    return au_bin_ent, au_bin_one_vs_rest_ent, au_bin_one_vs_rest_var, au_bin_var, au_entropy, au_variance, eu_bin_ent, eu_bin_one_vs_rest_ent, eu_bin_one_vs_rest_var, eu_bin_var, eu_entropy, eu_variance, tu_bin_ent, tu_bin_one_vs_rest_ent, tu_bin_one_vs_rest_var, tu_bin_var, tu_entropy, tu_variance


def get_single_estimator_predictions(X_test, clf, clf_name):
    y_pred_proba_estimators = []
    estimators = clf.estimators_
    for estimator in estimators:
        y_pred_proba = estimator.predict_proba(X_test)
        if clf_name == 'ENS (LGBM-GEO)':
            y_pred_proba = softmax(y_pred_proba, axis=1)
        y_pred_proba_estimators.append(y_pred_proba)
    estimator_results_stacked = np.stack(y_pred_proba_estimators, axis=2)
    return estimator_results_stacked
