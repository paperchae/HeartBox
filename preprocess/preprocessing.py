import os
import h5py

from read_database import *
from utils.mp_function import multi_process
from utils.config import get_config


def save_preprocessed_data(cfg, preprocessed_ecg, file_name, age) -> None:
    """
    Save preprocessed data into hdf5 file.
    :param cfg: configuration file (config.yaml)
    :param preprocessed_ecg: preprocessed ecg data
    :param file_name: index data of ecg data
    :param age: age data of ecg data
    :return: None
    """
    if cfg.debug:
        print("debug mode is on, not saving any data...")
        return
    else:  # save data
        root_dir = cfg.preprocess.save_path
        # set save path according to internal/external data
        if cfg.preprocess.data.internal:
            save_path = root_dir + "internal/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            mode = "train" if cfg.preprocess.data.train else "test"
            af = "af" if cfg.preprocess.data.af else "normal"
            if cfg.preprocess.option.augment:
                mode += "_augmented"
            dset_path = save_path + mode + "_" + af + ".hdf5"
        else:  # external data
            save_path = root_dir + "external/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            dset_path = save_path + "test.hdf5"
        print(len(file_name))

        # save data into hdf5 file ecg, file_name, label, age
        dset = h5py.File(dset_path, "w")
        dset["ecg"] = preprocessed_ecg
        dset["file_name"] = [int(x) for x in file_name]
        dset["label"] = (
            np.ones_like(file_name, dtype=float)
            if cfg.preprocess.data.af
            else np.zeros_like(file_name, dtype=float)
        )
        dset["age"] = age
        dset.close()


def read_total_data(cfg: Any) -> None:
    """
    Read index data and waveform data,$ preprocess them, and save them into hdf5 file.
    :param cfg: configuration file (config.yaml)
                cfg.preprocess.data
                cfg.preprocess.option
    :return: None
    """
    # read index data
    idx_df = read_idx_data(cfg.preprocess.data)

    # read waveform data & preprocess
    preprocessed_ecg, file_name, _, age = multi_process(
        target_function=read_waveform_data,
        patient_df=idx_df,
        preprocess_cfg=cfg.preprocess,
        debug_flag=cfg.debug,
    )

    # save preprocessed data
    save_preprocessed_data(cfg, preprocessed_ecg, file_name, age)


if __name__ == "__main__":

    # load configuration file for preprocessing
    config = get_config("../config.yaml")

    # to cope with data imbalance, augmentation is applied to True labeled data(Atrial Fibrillation, Atrial Flutter)
    if config.preprocess.all:
        # sweep all cases for training and testing
        #        [internal, train, af, augment]
        sweep = [
            [True, True, False, False],  # internal, train, normal, no augment
            [True, True, True, True],  # internal, train, af, augment
            [True, False, False, False],  # internal, test, normal, no augment
            [True, False, True, True],  # internal, test, af, augment
            [False, None, None, False],
        ]  # external, None, None, no augment
        for s in sweep:
            print(s)
            config.preprocess.data.internal = s[0]
            config.preprocess.data.train = s[1]
            config.preprocess.data.af = s[2]
            config.preprocess.option.augment = s[3]
            read_total_data(config)
    else:  # preprocess only one case based on config.yaml
        read_total_data(config)
