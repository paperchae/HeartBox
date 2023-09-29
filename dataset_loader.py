from typing import Any
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ECGDataset import ECGDataset, Compose, ToTensor, Standardize, RandomCrop, RandomMask, RandomVerticalFlip, \
    RandomHorizontalFlip, RandomNoise
import utils.visualization as vis


def dataset_loader(internal: bool = True,
                   load_augmented: bool = True,
                   split_ratio: float = 0.8,
                   train: bool = True,
                   batch_size: int = 512,
                   device: Any = None):
    """
    Load Dataset for Training, Validation, Test according to the parameters
    * Internal Train Dataset is divided into Train, Validation Dataset for Training
    ** While splitting the dataset, the age distribution is considered.
        By split original data into 10 groups according to the age, the train, validation dataset is divided.
        Ensuring that similar distribution of age is maintained for both train, validation dataset.

    :param internal: Flag for internal dataset
    :param load_augmented: Flag for loading augmented dataset
    :param split_ratio: Ratio for splitting train, validation dataset
    :param train: Flag for splitting train, validation dataset
    :param batch_size: Batch size for DataLoader
    :param device: Destination device for CustomDataset (ECGDataset)
    :return: train_loader, valid_loader, test_loader, external_test_loader
    """
    train_shuffle = True
    test_shuffle = False
    if internal:
        mode = 'train' if train else 'test'
        if load_augmented:
            dataset_root_path = 'data/preprocessed/internal/{}_augmented_af.hdf5'.format(mode)
        else:
            dataset_root_path = 'data/preprocessed/internal/{}_af.hdf5'.format(mode)
        dataset_root_path2 = 'data/preprocessed/internal/{}_normal.hdf5'.format(mode)
    else:  # external data
        dataset_root_path = 'data/preprocessed/external/test.hdf5'
        dataset_root_path2 = None

    with h5py.File(dataset_root_path, 'r') as true_data:
        ecg_af, label_af, age_af, idx_af = np.array(true_data['ecg']), \
                                           np.array(true_data['label']), \
                                           np.array(true_data['age']), \
                                           np.array(true_data['file_name'])
    if dataset_root_path2 is not None:
        with h5py.File(dataset_root_path2, 'r') as false_data:
            ecg_normal, label_normal, age_normal, idx_normal = np.array(false_data['ecg']), \
                                                               np.array(false_data['label']), \
                                                               np.array(false_data['age']), \
                                                               np.array(false_data['file_name'])
        ecg = np.concatenate((ecg_af, ecg_normal), axis=0)
        label = np.concatenate((label_af, label_normal), axis=0)
        age = np.concatenate((age_af, age_normal), axis=0)
        # round age to 10 & replace nan to 60
        age = np.nan_to_num(np.round(age, -1), nan=60)
        idx = np.concatenate((idx_af, idx_normal), axis=0)

        ecg, label, idx, age = dataset_shuffler(ecg, label, idx, age)
    else:
        ecg, label, idx, age = dataset_shuffler(ecg_af, label_af, idx_af, age_af)

    if train:
        train_ecg, valid_ecg = [], []
        train_idx, valid_idx = [], []
        train_label, valid_label = [], []
        train_age, valid_age = [], []
        for i, a in enumerate(np.arange(0, 10) * 10):
            if i != 9:
                condition = np.logical_and(age >= a, age < a + 10)
            else:
                condition = age >= a
            tr_ecg, val_ecg = split_train_valid(ecg[condition], split_ratio, True)
            train_ecg.append(tr_ecg)
            valid_ecg.append(val_ecg)

            temp = np.vstack((label[condition], age[condition], idx[condition]))
            tr_temp, val_temp = split_train_valid(temp, split_ratio, False)
            train_label.append(tr_temp[0])
            train_age.append(tr_temp[1])
            train_idx.append(tr_temp[2])
            valid_label.append(val_temp[0])
            valid_age.append(val_temp[1])
            valid_idx.append(val_temp[2])

        train_ecg, valid_ecg = np.concatenate(train_ecg, axis=0), np.concatenate(valid_ecg, axis=0)
        train_label, valid_label = np.concatenate(train_label, axis=0), np.concatenate(valid_label, axis=0)
        train_age, valid_age = np.concatenate(train_age, axis=0), np.concatenate(valid_age, axis=0)
        train_idx, valid_idx = np.concatenate(train_idx, axis=0), np.concatenate(valid_idx, axis=0)

        vis.train_validation_dataset_distribution(train_age, train_label, valid_age, valid_label)

        train_dataset = ECGDataset(train_ecg, train_label, train_idx, device,
                                   transform=Compose([ToTensor('float'), Standardize()]))
        valid_dataset = ECGDataset(valid_ecg, valid_label, valid_idx, device,
                                   transform=Compose([ToTensor('float'), Standardize()]))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=train_shuffle)

        return train_loader, valid_loader
    else:  # test
        test_dataset = ECGDataset(ecg, label, idx, device,
                                  transform=Compose([ToTensor('float'), Standardize()]))

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)

        return test_loader


def dataset_shuffler(ecg_data: np.ndarray,
                     label_data: np.ndarray,
                     idx_data: np.ndarray,
                     age_data: np.ndarray):
    """
    Shuffles the datasets in same order

    :param ecg_data: preprocessed ECG data
    :param label_data: label data corresponding to ECG data
    :param idx_data: discriminator
    :param age_data: age data corresponding to ECG data
    :return:
    """
    shuffle_array = np.arange(len(ecg_data))
    np.random.seed(122)
    np.random.shuffle(shuffle_array)

    return ecg_data[shuffle_array], label_data[shuffle_array], idx_data[shuffle_array], age_data[shuffle_array]


def split_train_valid(total_df: pd.DataFrame,
                      split_ratio: float,
                      is_ecg: bool):
    """
    Splits the dataset into train and validation set in 80:20 ratio
    :param total_df:
    :param split_ratio:
    :param is_ecg:
    :return:
    """
    if is_ecg:
        return total_df[:int(len(total_df) * split_ratio)], total_df[int(len(total_df) * split_ratio):]
    else:
        return total_df[:, :int(len(total_df[0]) * split_ratio)], total_df[:, int(len(total_df[0]) * split_ratio):]


# def contrastive_loader():
#     # TODO:
#     internal_train_true = 'data/preprocessed/internal/train_augmented_af.hdf5'
#     internal_train_false = 'data/preprocessed/internal/train_augmented_normal.hdf5'
#     internal_test_true = 'data/preprocessed/internal/test_af.hdf5'
#     internal_test_false = 'data/preprocessed/internal/test_normal.hdf5'
#
#     with h5py.File(internal_train_true, 'r') as internal_train_t:
#         ecg_train_true = np.array(internal_train_t['ecg'])
#         label_train_true = np.array(internal_train_t['label'])
#         age_train_true = np.array(internal_train_t['age'])
#         idx_train_true = np.array(internal_train_t['file_name'])
#     with h5py.File(internal_train_false, 'r') as internal_train_f:
#         ecg_train_false = np.array(internal_train_f['ecg'])
#         label_train_false = np.array(internal_train_f['label'])
#         age_train_false = np.array(internal_train_f['age'])
#         idx_train_false = np.array(internal_train_f['file_name'])
#     with h5py.File(internal_test_true, 'r') as internal_test_t:
#         ecg_test_true = np.array(internal_test_t['ecg'])
#         label_test_true = np.array(internal_test_t['label'])
#         age_test_true = np.array(internal_test_t['age'])
#         idx_test_true = np.array(internal_test_t['file_name'])
#     with h5py.File(internal_test_false, 'r') as internal_test_f:
#         ecg_test_false = np.array(internal_test_f['ecg'])
#         label_test_false = np.array(internal_test_f['label'])
#         age_test_false = np.array(internal_test_f['age'])
#         idx_test_false = np.array(internal_test_f['file_name'])
#
#     ecg = np.concatenate((ecg_train_true, ecg_train_false, ecg_test_true, ecg_test_false), axis=0)
#     label = np.concatenate((label_train_true, label_train_false, label_test_true, label_test_false), axis=0)
#     age = np.concatenate((age_train_true, age_train_false, age_test_true, age_test_false), axis=0)
#     idx = np.concatenate((idx_train_true, idx_train_false, idx_test_true, idx_test_false), axis=0)
#
#     ecg, label, idx, age = dataset_shuffler(ecg, label, idx, age)


if __name__ == '__main__':
    dataset_loader(internal=True, load_augmented=True, split_ratio=0.8, train=True, batch_size=512, device='cuda')
