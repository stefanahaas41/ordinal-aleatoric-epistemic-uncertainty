import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin


class LGBMUnimodalSoftLabelClassifier(BaseEstimator, ClassifierMixin):
    ALPHA = 0.1

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.classifier = LGBMClassifier(random_state=42,
                                         objective=LGBMUnimodalSoftLabelClassifier.unimodal_label_smoothing_loss)

    def fit(self, X, y):
        LGBMUnimodalSoftLabelClassifier.ALPHA = self.alpha
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    @staticmethod
    def geometric_smoothing(p, alpha):
        if alpha == 0:
            return p
        y = np.argmax(p)
        G = sum([(pow(alpha, abs(y - k)) * (1 - alpha)) if k != y else 0 for k in range(0, len(p))])

        for k, value in enumerate(p):
            if k == y:
                p[k] = 1 - alpha
            else:
                p[k] = 1 / G * pow(alpha, (abs(y - k) + 1)) * (1 - alpha)
        return p

    @staticmethod
    def get_geometric_soft_labels(classes, alpha):
        one_hot_targets = np.eye(classes)
        return np.apply_along_axis(lambda x: LGBMUnimodalSoftLabelClassifier.geometric_smoothing(x, alpha), axis=1,
                                   arr=one_hot_targets)

    @staticmethod
    def unimodal_label_smoothing_loss(y_true, y_pred):
        # Get unimodal soft labels
        classes = len(np.unique(y_true))
        soft_labels = LGBMUnimodalSoftLabelClassifier.get_geometric_soft_labels(classes,
                                                                                LGBMUnimodalSoftLabelClassifier.ALPHA)

        eps = 1e-6
        y_pred = y_pred.reshape((y_true.size, -1), order='F')
        num_rows, num_class = y_pred.shape
        prob = softmax(y_pred, axis=1)

        # Create one-hot encoding for true labels
        one_hot_targets = np.eye(num_class)[y_true.astype(int)]
        one_hot_targets_smoothed = soft_labels[np.argmax(one_hot_targets, axis=1)]

        grad = prob - one_hot_targets_smoothed
        factor = num_class / (num_class - 1)
        hess = factor * prob * (1 - prob)
        hess = np.where(hess > eps, hess, eps)
        grad = grad.ravel(order='F')
        hess = hess.ravel(order='F')
        return grad, hess
