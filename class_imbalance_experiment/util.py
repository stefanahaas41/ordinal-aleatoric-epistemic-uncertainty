import numpy as np
import pandas as pd
from dlordinal.metrics import amae, mmae, ranked_probability_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, brier_score_loss
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
    cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from benchmark_experiments.script import get_single_estimator_predictions, get_uncertainties
from util.uncertainty.rejection_curve import rejection_curve


from matplotlib import pyplot as plt
import seaborn as sns



def plot_data_dist(p, title):
    sns.set_theme(context='paper', style='ticks', font_scale=4, font="serif", rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsmath}'
    })


    palette = sns.color_palette("Blues_d")

    # Create a color list based on the highlight_classes
    colors = []
    for i in range(len(p)):

        colors.append(palette[i % len(palette)])  # Use the palette color for normal classes


    g = sns.barplot(x=[1,2,3,4,5],y=p,legend=False,palette=colors)
    g.set_yticks([0,2000,4000,6000,8000])
    g.set_yticklabels(["0", "2,000", "4,000", "6,000", "8,000"])


    g.set(xlabel='Class', ylabel="\\# Instances", xticks = range(0,len(p)) ,xticklabels=[1,2,3,4,5])
    plt.savefig(f'{title}.pdf', dpi=300, bbox_inches='tight')

def multiclass_imbalance_ratio(y):
    """
    For multiclass, return the ratio of the smallest class count to the largest class count.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain at least two classes")
    _, counts = np.unique(y, return_counts=True)
    return counts.max()  / counts.min()

def is_unimodal(probs):
    """Check if a 1D array is unimodal (increases to a peak, then decreases)."""
    peak_idx = np.argmax(probs)
    # Increasing up to peak
    inc = np.all(np.diff(probs[: peak_idx + 1]) >= 0)
    # Decreasing after peak
    dec = np.all(np.diff(probs[peak_idx:]) <= 0)
    return inc and dec


def check_unimodality(y_pred):
    """Check unimodality for each row in y_pred and return the proportion."""
    unimodal_flags = np.array([is_unimodal(row) for row in y_pred])
    # Proportion of rows that are unimodal
    proportion = np.mean(unimodal_flags)
    print(
        f"Unimodal predictions: {np.sum(unimodal_flags)} / {len(y_pred)} ({proportion})"
    )
    return proportion


def run_experiment(X, y, le, type):
    model = VotingClassifier(estimators=[
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
    ], voting='soft')

    # Initialize 10-fold stratified CV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    balanced_accuracies = []
    maes = []
    amaes = []
    mmaes = []
    qwks = []
    conf_matrix = []
    umods = []
    rpss = []
    bss = []
    nlls = []

    overall_result_rejection_curve = {
        "Rejection": [],
        "Performance": [],
        "Metric": [],
        "Measure": []
    }

    result = []


    # Define which classes ong to which group
    head = middle = tail = []
    match type:
        case "bimodal":
            head = [1,5]
            middle = []
            tail = [2,3,4]
        case "unimodal":
            head = [3]
            middle = []
            tail = [1,2,4,5]
        case "left_tail":
            head = [5]
            middle = []
            tail = [1,2,3,4]
        case "right_tail":
            head = [1]
            middle = []
            tail = [2,3,4,5]



    fold = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train LightGBM
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Predict
        estimator_results_stacked = get_single_estimator_predictions(X_test, model, '')
        y_pred_proba = model.predict_proba(X_test)

        # Evaluate performance
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        amae_val = amae(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        mmae_val = mmae(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        y_test_enc = le.transform(y_test)

        umod = check_unimodality(y_pred_proba)
        rps = ranked_probability_score(y_test_enc, y_pred_proba)
        nll = log_loss(y_test_enc, y_pred_proba)
        bs = brier_score_loss(y_test_enc, y_pred_proba)

        print(f"\nFold {fold} Accuracy: {accuracy:.2f}")
        print(f"Confusion Matrix (Fold {fold}):\n{cm}")

        balanced_accuracies.append(balanced_accuracy)
        accuracies.append(accuracy)
        maes.append(mae)
        amaes.append(amae_val)
        mmaes.append(mmae_val)
        qwks.append(qwk)
        umods.append(umod)
        rpss.append(rps)
        nlls.append(nll)
        bss.append(bs)

        if  len(conf_matrix) == 0:
            conf_matrix = cm
        else:
            conf_matrix = conf_matrix + cm

        # Test how well the method works per group
        for group in ["tail","full", "head","middle"]:

            if group == "head":
                mask = np.isin(y_test, head)
            elif group == "tail":
                mask = np.isin(y_test, tail)
            elif group == "middle":
                if len(middle) == 0:
                    continue
                mask = np.isin(y_test, middle)
            else:
                mask = np.full(y_test.shape, True)

            # Select sub-groups
            y_test_group = y_test[mask]
            y_pred_proba_group = y_pred_proba[mask]
            estimator_results_stacked_group = estimator_results_stacked[mask]


            # Get uncertainties
            (au_bin_ent, au_bin_one_vs_rest_ent, au_bin_one_vs_rest_var, au_bin_var,
             au_entropy, au_variance, eu_bin_ent, eu_bin_one_vs_rest_ent, eu_bin_one_vs_rest_var,
             eu_bin_var, eu_entropy, eu_variance, tu_bin_ent, tu_bin_one_vs_rest_ent, tu_bin_one_vs_rest_var,
             tu_bin_var, tu_entropy, tu_variance) = get_uncertainties(
                estimator_results_stacked_group, y_pred_proba_group)

            # Get PRR values and rejection curves
            rejection_curves, prr, _ = rejection_curve(['ACC', 'MAE', 'MSE'], {
                'Random': '',
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

            }, y_pred_proba_group, y_test_group, le)

            overall_result_rejection_curve["Rejection"].extend(rejection_curves["Rejection"])
            overall_result_rejection_curve["Performance"].extend(rejection_curves["Performance"])
            overall_result_rejection_curve["Measure"].extend(rejection_curves["Measure"])
            overall_result_rejection_curve["Metric"].extend(rejection_curves["Metric"])

            result.append(['Entropy', 'TU', prr['ACC'][0], prr['MAE'][0], prr['MSE'][0], group])
            result.append(['Entropy', 'EU', prr['ACC'][1], prr['MAE'][1], prr['MSE'][1], group])
            result.append(['Entropy', 'AU', prr['ACC'][2], prr['MAE'][2], prr['MSE'][2], group])
            result.append(['Binary Variance', 'TU', prr['ACC'][3], prr['MAE'][3], prr['MSE'][3], group])
            result.append(['Binary Variance', 'EU', prr['ACC'][4], prr['MAE'][4], prr['MSE'][4], group])
            result.append(['Binary Variance', 'AU', prr['ACC'][5], prr['MAE'][5], prr['MSE'][5], group])
            result.append(['Binary Entropy', 'TU', prr['ACC'][6], prr['MAE'][6], prr['MSE'][6], group])
            result.append(['Binary Entropy', 'EU', prr['ACC'][7], prr['MAE'][7], prr['MSE'][7], group])
            result.append(['Binary Entropy', 'AU', prr['ACC'][8], prr['MAE'][8], prr['MSE'][8], group])
            result.append(['Variance', 'TU', prr['ACC'][9], prr['MAE'][9], prr['MSE'][9], group])
            result.append(['Variance', 'EU', prr['ACC'][10], prr['MAE'][10], prr['MSE'][10], group])
            result.append(['Variance', 'AU', prr['ACC'][11], prr['MAE'][11], prr['MSE'][11], group])
            result.append(
                ['Binary Variance One vs. Rest', 'TU', prr['ACC'][12], prr['MAE'][12], prr['MSE'][12], group])
            result.append(
                ['Binary Variance One vs. Rest', 'EU', prr['ACC'][13], prr['MAE'][13], prr['MSE'][13], group])
            result.append(
                ['Binary Variance One vs. Rest', 'AU', prr['ACC'][14], prr['MAE'][14], prr['MSE'][14], group])
            result.append(
                ['Binary Entropy One vs. Rest', 'TU', prr['ACC'][15], prr['MAE'][15], prr['MSE'][15], group])
            result.append(
                ['Binary Entropy One vs. Rest', 'EU', prr['ACC'][16], prr['MAE'][16], prr['MSE'][16], group])
            result.append(
                ['Binary Entropy One vs. Rest', 'AU', prr['ACC'][17], prr['MAE'][17], prr['MSE'][17], group])




        fold += 1

    cols = ['Uncertainty Measure', 'Uncertainty Type', 'PRR (MCR)', 'PRR (MAE)', 'PRR (MSE)', 'Group']

    df = pd.DataFrame(result, columns=cols)
    df.to_csv(f'prrs_{type}.csv', index=False)

    # Save rejection curves
    overall_result_df = pd.DataFrame(overall_result_rejection_curve)
    overall_result_df.to_csv(f'rejection_curves_{type}.csv', index=False)

    # Overall results
    results = {
        "Balanced Accuracy": {"mean": np.mean(balanced_accuracies), "std": np.std(balanced_accuracies)},
        "Accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
        "MAE": {"mean": np.mean(maes), "std": np.std(maes)},
        "AMAE": {"mean": np.mean(amaes), "std": np.std(amaes)},
        "MMAE": {"mean": np.mean(mmaes), "std": np.std(mmaes)},
        "QWK": {"mean": np.mean(qwks), "std": np.std(qwks)},
        "RPS": {"mean": np.mean(rpss), "std": np.std(rpss)},
        "NLL": {"mean": np.mean(nlls), "std": np.std(nlls)},
        "BS": {"mean": np.mean(bss), "std": np.std(bss)},
        "UMOD": {"mean": np.mean(umods), "std": np.std(umods)}
    }


    print(conf_matrix)

    # Print results
    for metric, stats in results.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
