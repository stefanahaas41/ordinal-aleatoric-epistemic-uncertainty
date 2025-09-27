import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, auc

from util.measures.empirical_risk import empirical_risk


def rejection_curve(metrics, u_measures, y_pred_proba, y_test, le):

    result = {
        "Rejection": [],
        "Performance": [],
        "Metric": [],
        "Measure": []
    }

    prr = {}

    prr_plots = {}

    samples = len(y_pred_proba)

    # Get prediction depending on empirical risk
    y_pred_acc = le.inverse_transform(np.argmax(y_pred_proba, axis=1))
    y_pred_mae = le.inverse_transform(empirical_risk(y_pred_proba, loss="l1"))
    y_pred_mse = le.inverse_transform(empirical_risk(y_pred_proba, loss="l2"))

    # Get base errors
    base_error_acc = 1 - np.sum(y_test == y_pred_acc) / len(y_pred_acc)
    base_error_mae = np.sum(np.abs(y_test - y_pred_mae)) / len(y_pred_mae)
    base_error_mse = np.sum((y_test - y_pred_mse) ** 2) / len(y_pred_mse)

    # Determine MAE and MSE to later select largest deviations first
    orc_mae = np.abs(y_test - y_pred_mae)
    orc_mse = (y_test - y_pred_mse) ** 2

    for metric in metrics:  # ACC, MAE, MSE, etc.

        prr_plots[metric] = {}

        # PRR for metric
        prr[metric] = []

        for measure_name, measure_result in u_measures.items():  # Entropy, Variance, etc.

            if measure_name != 'Random':
                prr_plots[metric][measure_name] = {}

            percentages = []
            random = []
            orc = []
            errors = []

            for i in range(samples):

                rejection = (i / samples) * 100

                percentages.append(rejection)

                # Get largest uncertainties
                idx_uncertainty = np.argsort(measure_result)[::-1][:i]

                # Select un-rejected indices
                mask_uncertainty = np.ones(samples, dtype=bool)
                mask_uncertainty[idx_uncertainty] = False

                # Random
                random_index = np.random.choice(len(y_test), i, replace=False)
                mask_rand = np.ones(y_test.size, dtype=bool)
                mask_rand[random_index] = False

                if metric == "ACC":
                    # Get result for unrejected
                    result_uncertainty = y_pred_acc[mask_uncertainty]

                    metric_result = accuracy_score(y_test[mask_uncertainty], result_uncertainty)

                    errors.append(
                        (np.sum(y_test[mask_uncertainty] != result_uncertainty) / samples) / base_error_acc)

                    # Oracle
                    orc.append((base_error_acc - min((rejection / 100.0) / base_error_acc,
                                                     1) * base_error_acc) / base_error_acc)
                    # Random rejection
                    random.append((base_error_acc - (rejection / 100.0) * base_error_acc) / base_error_acc)

                    base_error = base_error_acc
                elif metric == "MAE":
                    # Get result for unrejected
                    result_uncertainty = y_pred_mae[mask_uncertainty]

                    metric_result = mean_absolute_error(y_test[mask_uncertainty], result_uncertainty)

                    # PRR
                    mae_unc = np.sum(np.abs(y_test[mask_uncertainty] - result_uncertainty))
                    errors.append((mae_unc / samples) / base_error_mae)

                    # Oracle rejects ideally
                    idx_orc_mae = np.argsort(orc_mae)[::-1][:i]
                    mask_orc_mae = np.ones(samples, dtype=bool)
                    mask_orc_mae[idx_orc_mae] = False
                    orc_mae_metric = np.sum(np.abs(y_test[mask_orc_mae] - y_pred_mae[mask_orc_mae])) / samples
                    orc.append(orc_mae_metric / base_error_mae)

                    # Random rejection
                    random.append((base_error_mae - (rejection / 100.0) * base_error_mae) / base_error_mae)

                    base_error = base_error_mae
                elif metric == "MSE":
                    # Get result for unrejected
                    result_uncertainty = y_pred_mse[mask_uncertainty]

                    metric_result = mean_squared_error(y_test[mask_uncertainty], result_uncertainty)
                    # PRR
                    mse_unc = np.sum((y_test[mask_uncertainty] - result_uncertainty) ** 2)
                    errors.append((mse_unc / samples) / base_error_mse)

                    # Oracle rejects ideally
                    idx_orc_mse = np.argsort(orc_mse)[::-1][:i]
                    mask_orc_mse = np.ones(samples, dtype=bool)
                    mask_orc_mse[idx_orc_mse] = False
                    orc_mse_metric = np.sum((y_test[mask_orc_mse] - y_pred_mse[mask_orc_mse]) ** 2) / samples
                    orc.append(orc_mse_metric / base_error_mse)

                    # Random rejection
                    random.append((base_error_mse - (rejection / 100.0) * base_error_mse) / base_error_mse)

                    base_error = base_error_mse

                # Assemble result for rejection curve
                result["Rejection"].append(rejection)
                result["Performance"].append(metric_result)
                result["Measure"].append(measure_name)
                result["Metric"].append(metric)

            if measure_name != 'Random':

                # Assemble prediction rejection ratio (PRR) for metric
                percentages = np.array(percentages)
                orc = np.array(orc)
                random = np.array(random)

                auc_uns = 1.0 - auc(percentages / 100.0, errors)

                auc_orc = 1.0 - auc(percentages / 100.0, orc)
                auc_rnd = 1.0 - auc(percentages / 100.0, random)

                print("Metric: " + metric + ", Measure: " + measure_name)
                print("Random: " + str(auc_rnd))
                print("ORC: " + str(auc_orc))
                print("Measure: " + str(auc_uns))

                prr_uns = ((auc_uns - auc_rnd) / (auc_orc - auc_rnd))
                print("PRR: " + str(prr_uns))
                prr[metric].append(prr_uns)

                # Save PRR plot data
                if prr_plots[metric] and prr_plots[metric][measure_name]:
                    prr_plots[metric][measure_name]['Rejection'].extend(percentages.tolist() +
                                                                        percentages.tolist() + percentages.tolist())
                    prr_plots[metric][measure_name]['Performance'].extend((random * base_error).tolist() +
                                                                          (np.array(errors) * base_error).tolist() + (
                                                                                  orc * base_error).tolist(), )
                    prr_plots[metric][measure_name]['Type'].extend(['Random' for i in range(len(percentages))] +
                                                                   [measure_name for i in range(len(percentages))] +
                                                                   ['ORC' for i in range(len(percentages))])
                else:
                    prr_plots[metric][measure_name] = {
                        'Rejection': percentages.tolist() + percentages.tolist() + percentages.tolist(),
                        'Performance': (random * base_error).tolist() + (np.array(errors) * base_error).tolist() + (
                                orc * base_error).tolist(),
                        'Type': ['Random' for _ in range(len(percentages))] +
                                [measure_name for _ in range(len(percentages))] +
                                ['ORC' for _ in range(len(percentages))]
                    }

    return result, prr, prr_plots


def plot_rejection_curve_df(df, errorbar, metric, out, title, mlflow_log=False):
    measures = df["Type"].unique()
    colors = sns.color_palette()

    custom_palette = {"Random": 'grey'}

    for i, measure in enumerate(measures):
        if measure != "Random":
            custom_palette[measure] = colors[i]
    # Round
    df['Rejection'] = df['Rejection'].round(0)
    # Plot
    plt.figure()
    g = sns.lineplot(data=df, x="Rejection", y="Performance", hue="Type", style="Type", errorbar=errorbar,
                     palette=custom_palette)
    g.set(xlabel='Rejection in %', ylabel=metric, title=title)
    plt.gca().legend().set_title('')
    plt.savefig(out, dpi=300, bbox_inches='tight')
