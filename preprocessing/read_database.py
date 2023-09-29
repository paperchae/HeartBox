from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
# from skimage.restoration import denoise_wavelet

from Handler import Manipulation, Cleansing
from signal_augmentation import Augmentation


def read_idx_data(data_cfg: Any) -> pd.DataFrame:
    """
    Read index data from csv file
    Split the data into AFIB_OR_AFL & normal

    :param data_cfg: configuration file (config.yaml)
    :return: either AFIB_OR_AFL or normal data

    """
    if data_cfg.internal:
        temp = 'train' if data_cfg.train else 'test'
        data_dir = '../data/index/internal/{}_index.csv'.format(temp)

        patient_df = pd.read_csv(data_dir, delimiter=',', index_col=0)
        df = patient_df.loc[patient_df['AFIB_OR_AFL'] == data_cfg.af]

    else:  # external data
        data_dir = '../data/index/external/external_test_index.csv'
        patient_df = pd.read_csv(data_dir, delimiter=',', index_col=0)
        df = patient_df

    return df


def read_waveform_data(i: int,
                       df: pd.DataFrame,
                       ecg_per_process: Any,
                       fname_per_process,
                       sex_per_process,
                       age_per_process,
                       preprocess_cfg,
                       debug_flag) -> None:
    """
    * If datasets are highly imbalanced for binary classification, Data Augmentation is highly recommended.
    Read waveform data from npy file with Dataframe from read_idx_data()
    If augment is True, preprocessed data will be augmented by 4 times

    Default preprocessing steps are:
        1. DownSample Signal from 500, 250Hz to 250Hz to match length
        2. Remove Baseline wandering by using bandpass filter [0, 0.5]Hz
        3. DownSample Signal from 250Hz to 125Hz for whole data utilization
        4. Denoise Signal by using wavelet denoising
        5. Normalize Signal by using min-max normalization for model to learn easily & match UOM of internal & external

    Data Augmentation methods are as followed:
        Using same functions as above, but with different combination of them
        1. down sample -> down sample -> normalize
        2. down sample -> baseline wander removal -> down sample -> normalize
        3. down sample -> denoising -> down sample -> normalize

    :param i:
    :param df:
    :param ecg_per_process:
    :param fname_per_process:
    :param sex_per_process:
    :param age_per_process:
    :param preprocess_cfg:
    :param debug_flag:
    :return:
    """
    waveform_dir = '../data/waveform/internal/{}' if preprocess_cfg.data.internal else '../data/waveform/external/{}'
    debug_cnt = 0
    # read waveform data
    waveform, filename, sex, age = [], [], [], []

    for f, sr, s, a in tqdm(zip(df['FILE_NAME'], df['SAMPLE_RATE'], df['SEX'], df['AGE']), desc='Process {}'.format(i),
                            leave=True, total=len(df)):
        if debug_flag:
            if debug_cnt == 10:
                break
            debug_cnt += 1
        ecg = np.load(waveform_dir.format(f))
        # down sampling 1 (from 500, 250Hz to 250Hz)
        ecg = Manipulation(ecg).down_sample(from_fs=sr, to_fs=250).squeeze()
        waveform.append(ecg)
        filename.append(f.split('_')[-1].split('.')[0])
        sex.append(s)
        age.append(a)

    waveform = np.array(waveform)

    # baseline wander removal
    base_ecg = Cleansing(waveform, fs=250, cpu=True).detrend(mode=preprocess_cfg.option.detrend)
    # down sampling 2 (from 250Hz to 125Hz)
    down_ecg = Manipulation(base_ecg).down_sample(from_fs=250, to_fs=preprocess_cfg.option.target_fs)
    # denoising
    # denoise_ecg = [denoise_wavelet(ecg, method='VisuShrink', mode=preprocess_cfg.option.denoise,
    #                                wavelet_levels=3, wavelet='sym9', rescale_sigma=True) for ecg in np.array(down_ecg)]
    # denoise_ecg = np.array(denoise_ecg)
    denoise_ecg = Cleansing(down_ecg).noise_removal(mode=preprocess_cfg.option.denoise)
    # normalize
    norm_ecg = Manipulation(denoise_ecg).normalize(preprocess_cfg.option.normalize)

    # default (w/o augmentation)
    ecg_per_process.append(np.array(norm_ecg))
    fname_per_process.append(filename)
    sex_per_process.append(sex)
    age_per_process.append(age)

    # data augmentation
    if preprocess_cfg.option.augment:
        # # augment1 : down sample -> normalize
        # aug1 = Manipulation(waveform).down_sample(from_fs=250, to_fs=preprocess_cfg.option.target_fs)
        # aug1 = Manipulation(aug1).normalize(preprocess_cfg.option.normalize)
        # # augment2 : baseline wander removal -> down sample -> normalize
        # aug2 = Cleansing(waveform, fs=250, cpu=True).detrend(mode=preprocess_cfg.option.detrend)
        # aug2 = Manipulation(aug2).down_sample(from_fs=250, to_fs=preprocess_cfg.option.target_fs)
        # aug2 = Manipulation(aug2).normalize(preprocess_cfg.option.normalize)
        # # augment3 : denoising -> down sample -> normalize
        # # aug3 = [denoise_wavelet(ecg, method='VisuShrink', mode=preprocess_cfg.option.denoise,
        # #                         wavelet_levels=3, wavelet='sym9', rescale_sigma=True) for ecg in np.array(waveform)]
        # # aug3 = np.array(aug3)
        # aug3 = Cleansing(waveform).noise_removal(preprocess_cfg.option.denoise)
        # aug3 = Manipulation(aug3).down_sample(from_fs=250, to_fs=preprocess_cfg.option.target_fs)
        # aug3 = Manipulation(aug3).normalize(preprocess_cfg.option.normalize)

        # TODO: Verify Augmentation class is working properly.
        aug1 = Augmentation(waveform, 250, 125, preprocess_cfg).augment1()
        aug2 = Augmentation(waveform, 250, 125, preprocess_cfg).augment2()
        aug3 = Augmentation(waveform, 250, 125, preprocess_cfg).augment3()

        ecg_per_process.append(np.array(aug1))
        ecg_per_process.append(np.array(aug2))
        ecg_per_process.append(np.array(aug3))
        for _ in range(3):
            fname_per_process.append(filename)
            sex_per_process.append(sex)
            age_per_process.append(age)
