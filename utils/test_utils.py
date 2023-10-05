import os
import datetime
import pandas as pd
from utils.visualization import (
    hist_inference_data_by_age,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_sensitivity_verse_specificity,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from utils.metrics import Metrics


def save_inference_result(
    external_test_prob: iter, external_test_prediction: iter, external_test_index: iter
) -> None:
    """
    1. Save inference result in data/index/external/external_test_index.csv using index
    2. Plot histogram of inference data by age

    :param external_test_prob: probability of AFIB_OR_AFL
    :param external_test_prediction: 1 if AFIB_OR_AFL else 0
    :param external_test_index: index of external test data
    :return: None
    """
    external_dataframe = pd.read_csv(
        "data/index/external/external_test_index.csv", index_col=0
    )
    for prob, pred, index in zip(
        external_test_prob, external_test_prediction, external_test_index
    ):
        index = str(int(index)).zfill(5)
        file_name = "external_waveform_" + index + ".npy"
        external_dataframe.loc[
            external_dataframe["FILE_NAME"] == file_name,
            ["AFIB_OR_AFL_PROB", "AFIB_OR_AFL"],
        ] = [float(prob), bool(pred)]
    external_dataframe.to_csv("data/index/external/external_test_index_tempp.csv")
    print("Inference result saved in data/index/external/external_test_index_tempp.csv")
    hist_inference_data_by_age()
    print("test")


def save_metric_result(cfg, results) -> None:
    """
    Save metric result in dataframe using datetime as index

    :param cfg: config containing metric names, save path
    :param results: evaluated metric results
    :return: None
    """
    if cfg.debug:
        pass
    results = [round(result, 5) for result in results]
    csv_file = cfg.test.save_path + "metric_result.csv"
    idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # index of result

    new_result = pd.DataFrame(columns=cfg.test.metrics, index=[idx])
    new_result[cfg.test.metrics] = [results]
    if os.path.isfile(csv_file):
        remaining_result = pd.read_csv(csv_file, index_col=0)
        total_result = pd.concat([remaining_result, new_result]).sort_index()
    else:
        total_result = new_result
    total_result.to_csv(csv_file)
    print(f"Metric result saved in {csv_file}")


def compute_metrics(load_best_model, prob, pred, label, test_cfg) -> None:
    """
    Compute metrics of inference result & save metric result in dataframe
    Including AUROC, Accuracy, Precision, Recall, F1, Confusion Matrix, Specificity, Sensitivity

    :param load_best_model: True if load best model from pretrained model directory. Set False for Training.
    :param prob: probability of AFIB_OR_AFL
    :param pred: 1 if AFIB_OR_AFL else 0
    :param label: True label of Test Data
    :param test_cfg: config containing metric names, save path
    :return:
    """
    prob = prob.numpy()
    pred = pred.numpy()
    label = label.numpy()

    met = Metrics(prob, label, cutoff=test_cfg.threshold)

    # auroc = roc_auc_score(label, prob)
    auroc = met.auroc()
    # fpr, tpr, thresholds = roc_curve(label, prob)
    fpr, tpr = met.fpr, met.tpr

    # accuracy = accuracy_score(label, pred)
    accuracy = met.accuracy()
    # precision = precision_score(label, pred)
    precision = met.precision()
    # recall = recall_score(label, pred)
    recall = met.recall()
    # f1 = f1_score(label, pred)
    f1 = met.f1score()

    # confusion_mat = confusion_matrix(label, pred)
    # tn, fp, fn, tp = confusion_mat.ravel()
    tn, fp, fn, tp = met.tn, met.fp, met.fn, met.tp
    # specificity = tn / (tn + fp)
    # sensitivity = tp / (tp + fn)
    sensitivity, specificity = met.sens_and_spec_with_cutoff()

    # Plot Confusion Matrix
    # plot_confusion_matrix(confusion_mat)
    plot_confusion_matrix([[tn, fp], [fn, tp]])

    # Plot the ROC curve
    plot_roc_curve(fpr, tpr, auroc)

    # Plot sensitivity vs specificity
    plot_sensitivity_verse_specificity(prob, label)

    print(
        f"AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},"
        f" F1: {f1:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}"
    )

    # Save metric result
    # results = [auroc, sensitivity, specificity, accuracy, precision, recall, f1]
    results = []
    for metric in test_cfg.metrics:
        if metric == "AUROC":
            results.append(auroc)
        elif metric == "Sensitivity":
            results.append(sensitivity)
        elif metric == "Specificity":
            results.append(specificity)
        elif metric == "Accuracy":
            results.append(accuracy)
        elif metric == "Precision":
            results.append(precision)
        elif metric == "Recall":
            results.append(recall)
        elif metric == "F1":
            results.append(f1)
        else:
            raise ValueError(f"Invalid metric name: {metric}")
    save_metric_result(test_cfg, results)


# def compare_pred_label_external():
#     """
#     Read external_test_index.csv and external_test_index_gt.csv and compare AFIB_OR_AFL and AFIB_OR_AFL_GT
#     :return: None
#     """
#
#     external_dataframe = pd.read_csv('data/index/external/external_test_index.csv', index_col=0)
#     external_dataframe_gt = pd.read_csv('data/index/external/external_test_index_gt.csv', index_col=0)
#     prediction = external_dataframe['AFIB_OR_AFL']
#     prediction_true = external_dataframe['AFIB_OR_AFL'] == True
#     prediction_false = external_dataframe['AFIB_OR_AFL'] == False
#     probability = external_dataframe['AFIB_OR_AFL_PROB']
#     ground_truth = external_dataframe_gt['AFIB_OR_AFL_GT']
#     ground_truth_true = external_dataframe_gt['AFIB_OR_AFL_GT'] == True
#     ground_truth_false = external_dataframe_gt['AFIB_OR_AFL_GT'] == False
#
#     compute_metrics(load_best_model=False, prob=probability, pred=prediction, label=ground_truth,
#                     test_cfg=['AUROC', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
