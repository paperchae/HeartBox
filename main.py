import os
import datetime

import torch
import torch.nn as nn

from net1d import Net1D
from dataset_loader import dataset_loader
from train import train_one_epoch
from test import test_one_epoch, inference
from validation import validation_one_epoch

from utils.config import get_config
from utils.train_utils import guarantee_reproducibility, check_preprocessed_data, save_model, early_stopping
from utils.test_utils import save_inference_result, compute_metrics
from utils.visualization import plot_training_results
from utils.wandb_function import init, log_train_valid, log_test

"""
Set up GPU environment & lock seed for reproducibility
"""
DEVICE = guarantee_reproducibility(random_seed=125)


def main(cfg) -> None:
    """
    TODO: Main function for training and evaluating model.
        1. Set up dataloaders
        2. Set up model
        3. Train model
        4. Evaluate model
        5. Save results

    0. Preprocess data if not exist
    1. Load datasets ( internal train(0.8), internal validation(0.2), internal test, external test )
    2. Set up model ( Slightly Modified from the original model )
    3. Set up loss_fn, optimizer, scheduler ( BinaryCrossEntropyWithLogitsLoss, AdamW, ExponentialLR Respectively )
    4. Train model for 200 epochs ( if validation loss decrease, save and test model )
    5. Evaluate metrics ( AUROC, Sensitivity, Specificity, Accuracy, Precision, Recoll and F1 score )
    6. Save external test results ( probability: float, prediction results: boolean )

    :param cfg: Configurations for preprocessing, training, testing and debugging.
                 See "config.yaml" for more details


    Each Model training results should be saved in each directory:
        1. Saving model weights (best model) in "models" directory
        2. Saving Internal Test results in "result" directory
        3. External test results(output probability, prediction results(boolean)) in "result" directory
    """

    # Internal Train, Validation Dataset Loaders
    train_loader, valid_loader = dataset_loader(internal=True, load_augmented=cfg.train.general.load_augmented,
                                                split_ratio=cfg.train.general.split_ratio, train=True,
                                                batch_size=cfg.train.general.batch_size, device=DEVICE)
    # Internal Test Dataset Loader
    test_loader = dataset_loader(internal=True, load_augmented=cfg.train.general.load_augmented,
                                 train=False, batch_size=cfg.test.batch_size, device=DEVICE)

    # External Test Dataset Loader
    external_test_loader = dataset_loader(internal=False, load_augmented=False,
                                          train=False, batch_size=cfg.test.batch_size, device=DEVICE)

    # Set Model, Loss function, Optimizer, Scheduler
    model = Net1D(cfg=cfg).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=cfg.train.hyperparameter.lr,
                                  weight_decay=cfg.train.hyperparameter.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.train.hyperparameter.gamma)

    train_costs = []
    valid_costs = []
    test_costs = []
    save_points = []
    model_saved_epochs = []

    log_flag = True if not cfg.debug and cfg.wandb.flag else False
    init(cfg) if log_flag else None

    # Train, Validation, Test
    print('----- Start Training -----')

    for epoch in range(cfg.train.general.epochs):
        train_costs.append(train_one_epoch(epoch=epoch, model=model, loader=train_loader,
                                           optimizer=optimizer, scheduler=scheduler, loss=loss_fn,
                                           device=DEVICE, wandb=cfg.wandb.flag))
        valid_costs.append(validation_one_epoch(epoch=epoch, model=model, loader=valid_loader,
                                                loss=loss_fn, device=DEVICE, wandb=cfg.wandb.flag))
        # scheduler.step()  <- moved to train_one_epoch()
        log_train_valid(train_costs[-1], valid_costs[-1], epoch, cfg.train.general.loss_fn, log_flag)
        if early_stopping(valid_costs, cfg.train.general.early_stop_n):
            break
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_points.append(current_time)
        save_flag = save_model(valid_costs, save_points, model,
                               epoch=epoch, model_save_path=cfg.train.model_save_path)  # **kwargs
        if save_flag:
            test_costs.append(test_one_epoch(compute_metric=False, epoch=epoch, model=model, loader=test_loader,
                                             metrics=cfg.test.metrics, threshold=cfg.test.threshold, loss=loss_fn,
                                             device=DEVICE, wandb=cfg.wandb.flag))
            log_test(test_costs[-1], epoch, cfg.train.general.loss_fn, log_flag)
            if not cfg.debug:
                model_saved_epochs.append(epoch)

    # Evaluate Metrics & Save Internal Test Results
    print('----- Start Evaluation -----')
    prob, pred, label, index = test_one_epoch(compute_metric=True, epoch=cfg.train.general.epochs, model=model,
                                              loader=test_loader, metrics=cfg.test.metrics,
                                              threshold=cfg.test.threshold,
                                              loss=loss_fn, device=DEVICE, wandb=cfg.wandb.flag)
    compute_metrics(load_best_model=False, prob=prob, pred=pred, label=label, test_cfg=cfg)

    # Plot Training Results( train, validation, test / BCEwithLogitsLoss )
    if not cfg.wandb.flag:
        plot_training_results(train_costs, valid_costs, test_costs, model_saved_epochs)

    # External Dataset Test
    print('----- External Test -----')
    external_test_prob, external_test_prediction, external_test_index = inference(model=model,
                                                                                  loader=external_test_loader,
                                                                                  device=DEVICE)
    save_inference_result(external_test_prob, external_test_prediction, external_test_index)


if __name__ == "__main__":
    root_path = '/home/paperc/PycharmProjects/VUNO_HATIV_RECRUITING_PROJECT/data/MIMIC_ECG/'

    for roots, dirs, files in os.walk(root_path):
        for file in files:
            print('file:', file)
            if '.hea' in file:
                print('break')
            print('break')
    # Load Configurations
    config = get_config('config.yaml')

    # Check Data Existence
    check_preprocessed_data(config)

    # Train and Evaluate Model
    main(cfg=config)
