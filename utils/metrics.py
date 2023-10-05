import numpy as np
from typing import Any, Callable, Optional, Tuple


# file:///Users/paperc/Documents/afib_classification_springer.pdf
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8583162/pdf/ijerph-18-11302.pdf


class Metrics:
    def __init__(self, probs: np.ndarray, labels: np.ndarray, cutoff: float = 0.5):
        """
        Computes metrics for binary classification.

        :param probs:  array of probabilities.
        :param labels:  array of labels.
        :param cutoff:  cutoff value for binary classification.
        """
        self.probs = probs
        self.labels = labels
        self.cutoff = cutoff

        self.tp, self.tn, self.fp, self.fn = self.confusion_matrix()
        self.fpr, self.tpr = self.fpr_tpr()

    def confusion_matrix(self) -> Tuple[float, float, float, float]:
        """
        Computes the confusion matrix.
        Returns:
            Confusion matrix.
        """

        y_pred_threshold = np.where(self.probs > self.cutoff, 1, 0)
        tp = np.sum((y_pred_threshold == 1) & (self.labels == 1))
        tn = np.sum((y_pred_threshold == 0) & (self.labels == 0))
        fp = np.sum((y_pred_threshold == 1) & (self.labels == 0))
        fn = np.sum((y_pred_threshold == 0) & (self.labels == 1))

        return tp, tn, fp, fn

    def fpr_tpr(self) -> Tuple[float, float]:
        """
        Computes the false positive rate and true positive rate.

        Returns:
            False positive rate and true positive rate.
        """

        tp, tn, fp, fn = self.confusion_matrix()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        return fpr, tpr

    def accuracy(self) -> float:
        """
        Computes the accuracy.

        Returns:
            Accuracy.
        """

        y_pred_threshold = np.where(self.probs > self.cutoff, 1, 0)

        return np.sum(y_pred_threshold == self.labels) / len(self.labels)

    def precision(self) -> float:
        """
        Computes the precision.

        Returns:
            Precision.
        """

        precision = self.tp / (self.tp + self.fp)

        return precision

    def recall(self) -> float:
        """
        Computes the recall.

        Returns:
            Recall.
        """

        recall = self.tp / (self.tp + self.fn)

        return recall

    def auroc(self) -> float:
        """
        Computes the area under the receiver operating characteristic curve.

        Returns:
            Area under the receiver operating characteristic curve.
        """

        auroc = np.trapz(
            self.tpr, self.fpr
        )  # Integration using the composite trapezoidal rule (area under curve)

        return auroc

    def f1score(self) -> float:
        """
        Computes the F1 score.

        Returns:
            F1 score.
        """

        precision = self.precision()
        recall = self.recall()

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def sens_and_spec_with_cutoff(self) -> Tuple[float, float]:
        """
        Computes the sensitivity and specificity with a given cutoff.

        Returns:
            Sensitivity and specificity.
        """

        tp, tn, fp, fn = self.confusion_matrix()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return sensitivity, specificity
