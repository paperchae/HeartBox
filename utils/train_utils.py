import os
from typing import Any

import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

import numpy as np


def guarantee_reproducibility(random_seed: int = 125) -> Any:
    """
    Lock seed for reproducibility and reduce training time.

    Considerations:
        Seed:
            - numpy.random.seed()
            - torch.manual_seed()
            - torch.cuda.manual_seed()
            - torch.cuda.manual_seed_all() if multi-GPU
            - torch.backends.cudnn.deterministic = True
            - torch.backends.cudnn.benchmark = False
        Training speed:
            - cuda.allow_tf32 = True
            - cudnn.allow_tf32 = True

    See https://pytorch.org/docs/stable/notes/randomness.html for more information.

    :return: DEVICE
    """
    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('----- GPU INFO -----\nDevice:', DEVICE)
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())
    gpu_ids = list(map(str, list(range(torch.cuda.device_count()))))
    total_gpu_memory = 0
    for gpu_id in gpu_ids:
        total_gpu_memory += torch.cuda.get_device_properties("cuda:" + gpu_id).total_memory
    print('Total GPU Memory :', total_gpu_memory, '\n--------------------')

    # Lock seed for reproducibility
    if torch.cuda.is_available():
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        cuda.manual_seed(random_seed)
        cuda.allow_tf32 = True  # allowing TensorFloat32 for faster training
        # torch.cuda.manual_seed_all(random_seed)   # if use multi-GPU
        cudnn.enabled = True
        cudnn.deterministic = True  # turn on for reproducibility ( if turned on, slow down training )
        cudnn.benchmark = False  # turn on for faster training ( if turned on, may be not reproducible )
        cudnn.allow_tf32 = True  # allowing TensorFloat32 for faster training
    else:
        print("cuda is not available")
    return DEVICE


def best_model_selection(model_path: str) -> str:
    """
    Select the best model from the model list.
    * Not used in this project.

    :param model_path: root path of saved models
    :return: lowest cost model list name
    """
    model_list = os.listdir(model_path)
    cost_list = [model.split('_')[1] for model in model_list]

    return model_list[np.argmin(cost_list)]


def check_preprocessed_data(cfg) -> None:
    """
    Check if the preprocessed data exists.

    :param cfg: save path of preprocessed data
    :return: None
    """
    data_list = ['internal/test_normal.hdf5', 'internal/test_augmented_af.hdf5', 'internal/train_normal.hdf5',
                 'internal/train_augmented_af.hdf5', 'external/test.hdf5']
    for data in data_list:
        path = cfg.preprocess.save_path + data
        if os.path.exists(path[3:]):
            print('Preprocessed data found: {}'.format(path))
            continue
        else:
            print('Preprocessed data not found: {}'.format(path))
            print('Please run preprocessing.py first with config.preprocess.all = True')
            raise FileNotFoundError


def save_model(valid_costs: list,
               save_point: list,
               model,
               **kwargs) -> bool:
    """
    If the current model has the lowest validation cost, save the model and remove the prior model.
    *If current epoch is 0, does not save the model.

    :param valid_costs: validation cost list
    :param save_point: save point list (model save time)
    :param model: model to save
    :param kwargs: current epoch, model save path
    :return: True if the model is saved, False if not for plotting loss graph
    """
    if kwargs['epoch'] != 0:
        # if train_costs[-1] < min(train_costs[:-1]) and valid_costs[-1] < min(valid_costs[:-1]):
        if valid_costs[-1] < min(valid_costs[:-1]):
            save_path = kwargs['model_save_path'] + 'cost_{}_time_{}.pt'.format(valid_costs[-1], save_point[-1])
            torch.save(model.state_dict(), save_path)
            print('\nsaved model: {}'.format(save_path))
            if kwargs['epoch'] > 1:
                try:
                    prior_cost = min(valid_costs[:-1])
                    prior_path = kwargs['model_save_path'] + 'cost_{}_time_{}.pt'.format(prior_cost, save_point[-2])
                    os.remove(prior_path)
                    print('removed prior model: {}'.format(prior_path))
                except:
                    print('failed to remove prior model')
            return True
    else:
        return False


def early_stopping(valid_costs: list, n: int = 10) -> bool:
    """
    Early stopping if the validation cost is not decreased for 10 epochs.

    :param valid_costs: validation cost list
    :param n: number of epochs to check
    :return: True if early stopping, False if not
    """
    if len(valid_costs) > n:
        if min(valid_costs[-n:]) > min(valid_costs[:-n]):
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    best_model = best_model_selection(model_path='../models/')
    print(best_model)
