import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def get_process_num(segment_num: int) -> int:
    """
    Returns the number of process to be used for multiprocessing.
    Not recommended to use all the cores of the CPU.

    :param segment_num: total number of task(segment) to be preprocessed
    :return: process_num: number of process to be used
    """

    divisors = []
    for i in range(1, int(segment_num ** 0.5) + 1):
        if segment_num % i == 0:
            divisors.append(i)
            if i != segment_num // i:
                divisors.append(segment_num // i)
    available_divisor = [x for x in divisors if x < os.cpu_count()]

    return (
        int(os.cpu_count() * 0.6)
        if np.max(available_divisor) < os.cpu_count() // 2
        else np.max(available_divisor)
    )


def augmentation_integrity_checker(fname_total, sex_total, age_total) -> None:
    """
    To guarantee the order of the data when augmenting data using multiprocessing.
    It checks whether the data is augmented in the same order as the original data.
    * It seems that running multiprocess when high memory usage is required/used causes the disorder.
    ** Try multiprocess option off if you get a ValueError.

    :param fname_total:
    :param sex_total:
    :param age_total:
    :return:
    """
    wrong_idx = []
    for f in tqdm(
        list(set(fname_total)), desc="Checking Augmentation Index Integrity..."
    ):
        if len(list(set(sex_total[fname_total == f]))) != 1:
            # print(sex_total[fname_total == f])
            # print(fname_total[fname_total == f])
            if np.isnan(sex_total[fname_total == f]).all():
                pass
            else:
                wrong_idx.append(f)
        if len(list(set(age_total[fname_total == f]))) != 1:
            # print(age_total[fname_total == f])
            # print(fname_total[fname_total == f])
            if np.isnan(age_total[fname_total == f]).all():
                pass
            else:
                wrong_idx.append(f)
    if len(wrong_idx) == 0:
        print("*** Index Integrity Check Passed ***")
    else:
        print("Wrong Index: {}, Set preprocess.multi to False".format(wrong_idx))
        raise ValueError("*** Index Integrity Check Failed ***")


def multi_process(
    target_function, patient_df: pd.DataFrame, preprocess_cfg, debug_flag: bool = False
) -> tuple:
    """
    1. Split the patient_df into process_num
    2. Run target_function in parallel
    3. Concatenate the results from each process
    4. Check the integrity of the results
    5. Return ECG, FILE_NAME, SEX, AGE ( sex data is only used for integrity check )

    :param target_function: read_waveform_data
    :param patient_df: dataframe divided by process_num
    :param preprocess_cfg: preprocessing configuration
    :param debug_flag: flag for debugging
    :return:
    """
    if preprocess_cfg.multi and not debug_flag:
        process_num = get_process_num(len(patient_df))
    else:
        process_num = 1
    print("process_num: {}".format(process_num))

    target_length = preprocess_cfg.option.target_fs * preprocess_cfg.data.time
    patients_per_process = np.array_split(patient_df, process_num)

    with mp.Manager() as manager:
        start_time = time.time()

        ecg_per_process = manager.list()
        fname_per_process = manager.list()
        sex_per_process = manager.list()
        age_per_process = manager.list()

        workers = [
            mp.Process(
                target=target_function,
                args=(
                    process_i,
                    patients_per_process[process_i],
                    ecg_per_process,
                    fname_per_process,
                    sex_per_process,
                    age_per_process,
                    preprocess_cfg,
                    debug_flag,
                ),
            )
            for process_i in range(process_num)
        ]

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        ecg_total = np.concatenate(np.array(ecg_per_process), axis=0)
        fname_total = np.concatenate(fname_per_process)
        sex_total = np.concatenate(sex_per_process)
        age_total = np.concatenate(age_per_process)

        assert ecg_total.shape[-1] == target_length
        print("--- %s seconds ---" % (time.time() - start_time))
        manager.shutdown()

    augmentation_integrity_checker(fname_total, sex_total, age_total)

    return ecg_total, fname_total, sex_total, age_total
