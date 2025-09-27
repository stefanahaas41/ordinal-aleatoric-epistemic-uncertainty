import numpy as np

def risk_of_prediction(y_pred, y_pred_proba, loss="l1"):
    risk = np.zeros(y_pred_proba.shape[0])
    for i in range(y_pred_proba.shape[0]):
        for k in range(y_pred_proba.shape[1]):
            if loss == "l2":
                dist = np.abs(k - y_pred[i]) ** 2
            else:
                dist = np.abs(k - y_pred[i])
            risk[i] += dist * y_pred_proba[i, k]
    return risk



def empirical_risk(y_pred_proba, loss="l1"):
    y_pred = np.zeros(y_pred_proba.shape[0])
    for i in range(y_pred_proba.shape[0]):
        class_risks = np.zeros(y_pred_proba.shape[1])
        # Risk per class
        for k in range(y_pred_proba.shape[1]):
            # Other classes
            for j in range(y_pred_proba.shape[1]):
                if loss == "l2":
                    dist = np.abs(k - j) ** 2
                else:
                    dist = np.abs(k - j)
                class_risks[k] += dist * y_pred_proba[i, j]
        y_pred[i] = np.argmin(class_risks)
    return y_pred.astype(int)
