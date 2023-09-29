import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils.metrics import *
from preprocessing.generate_mimic_data import ECG


# Preprocessing
def plot_augmented_waveforms(clean, aug1, aug2, aug3, chunk: int = 500) -> None:
    """
    Plot Augmented Waveforms for comparison

    :param clean: all
    :param aug1: down sample -> down sample -> normalize
    :param aug2: down sample -> baseline wander removal -> down sample -> normalize
    :param aug3: down sample -> denoising -> down sample -> normalize
    :param chunk: length of chunk to plot
    :return: None
    """
    for i in range(10):
        plt.title('Data Augmentation')
        plt.plot(aug1[i][:chunk], label='aug1')
        plt.plot(aug2[i][:chunk], label='aug2')
        plt.plot(aug3[i][:chunk], label='aug3')
        plt.plot(clean[i][:chunk], label='total')
        plt.legend(loc='upper right')
        plt.show()


def train_validation_dataset_distribution(train_age, train_label, valid_age, valid_label) -> None:
    """
    Plot Histogram of Train/Validation Dataset Distribution
    * To Ensure that the train and validation dataset have similar distribution

    :param train_age: Age of train dataset which have True labels (AFIB or AFL)
    :param train_label: True label of train dataset
    :param valid_age: Age data of validation dataset which have True labels (AFIB or AFL)
    :param valid_label: True label of validation dataset
    :return: None
    """
    train_true = np.round(np.sum(train_label) / len(train_label) * 100, 2)
    valid_true = np.round(np.sum(valid_label) / len(valid_label) * 100, 2)
    plt.title('Train/Validation Dataset Distribution' +
              '\n\n' + 'Train True: ' + str(train_true) + '% Validation True: ' + str(valid_true) + '%')
    plt.hist(train_age, bins=10, alpha=0.5, label='train')
    plt.hist(valid_age, bins=10, alpha=0.5, label='valid')
    plt.legend(loc='upper right')
    plt.show()


def eetest_dataset_distribution(test_age, test_label):
    """

    :param test_age:
    :param test_label:
    :return: None
    """
    test_true = np.round(np.sum(test_label) / len(test_label) * 100, 2)
    plt.title('Test Dataset Distribution' + '\n' + 'Test True: ' + str(test_true) + '%')
    plt.hist(test_age, bins=10, alpha=0.5, label='test')
    plt.legend(loc='upper right')
    plt.show()


def plot_training_results(train_loss: list, valid_loss: list, test_loss: list, saved_epochs: list) -> None:
    """
    main.py
    Plot training results
    * Test loss is plotted when the model is saved

    :param train_loss: loss of train dataset
    :param valid_loss: loss of validation dataset
    :param test_loss: loss when the model is saved
    :param saved_epochs: epochs when the model is saved
    :return: None
    """
    plt.title('Training Results (Binary Cross Entropy Loss)')
    plt.plot(train_loss, 'g-', label='Train Loss')
    plt.plot(valid_loss, 'b--', label='Validation Loss')
    plt.plot(saved_epochs, test_loss, 'rx', label='Test Loss')
    plt.legend(loc='upper right')
    plt.show()


def hist_inference_data_by_age() -> None:
    """
    Compare histogram if Inference Data Distribution by Age with Train/Test Dataset Distribution
    :return: None
    """
    external_df = pd.read_csv('data/index/external/external_test_index_temp.csv', index_col=0)
    external_list = external_df[external_df['AFIB_OR_AFL'] == 1]['AGE'].to_numpy()
    internal_train_df = pd.read_csv('data/index/internal/train_index.csv', index_col=0)
    internal_train_list = internal_train_df[internal_train_df['AFIB_OR_AFL'] == 1]['AGE'].to_numpy()
    internal_test_df = pd.read_csv('data/index/internal/test_index.csv', index_col=0)
    internal_test_list = internal_test_df[internal_test_df['AFIB_OR_AFL'] == 1]['AGE'].to_numpy()

    plt.title('Inference Data Distribution by Age')
    plt.hist(external_list, bins=10, alpha=0.5, label='external')
    plt.hist(internal_train_list, bins=10, alpha=0.5, label='internal_train')
    plt.hist(internal_test_list, bins=10, alpha=0.5, label='internal_test')
    plt.legend(loc='upper right')
    plt.show()


# Evaluation
def plot_sensitivity_verse_specificity(prob: list, label: list) -> None:
    """
    Plot Sensitivity vs Specificity

    :param prob:  probability of prediction
    :param label:  true label
    :return: None
    """
    sensitivity_list = []
    specificity_list = []
    threshold_list = np.linspace(0.1, 0.9, 9)
    for threshold in threshold_list:
        sensitivity, specificity = Metrics(prob, label, threshold).sens_and_spec_with_cutoff()
        # y_pred_threshold = (prob >= threshold).astype(int)
        # true_positives = np.sum((y_pred_threshold == 1) & (label == 1))
        # true_negatives = np.sum((y_pred_threshold == 0) & (label == 0))
        # false_positives = np.sum((y_pred_threshold == 1) & (label == 0))
        # false_negatives = np.sum((y_pred_threshold == 0) & (label == 1))
        #
        # sensitivity = true_positives / (true_positives + false_negatives)
        # specificity = true_negatives / (true_negatives + false_positives)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    plt.title('Sensitivity vs Specificity')
    plt.plot(threshold_list, sensitivity_list, 'g-', label='Sensitivity')
    plt.plot(threshold_list, specificity_list, 'b--', label='Specificity')
    plt.xlabel('Threshold')
    plt.ylabel('Sensitivity/Specificity')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.show()


def plot_roc_curve(fpr, tpr, auroc) -> None:
    """
    Plot ROC Curve
    :param fpr: false positive rate
    :param tpr: true positive rate
    :param auroc: area under the roc curve
    :return: None
    """
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_confusion_matrix(confusion_matrix):
    plt.title('Confusion Matrix')
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_patient_by_study(waveform_path):
    mimic = ECG(waveform_path)
    plt.title('Patient distribution by ECG studies')
    plt.xlabel('Number of studies')
    plt.ylabel('Number of patients')
    study = np.arange(1, 100, 1)
    patient_list = []
    for i in study:
        patient_list.append(len(mimic.get_patient_over_n_study(study_num=i)))

    plt.plot(study, patient_list)
    plt.show()
