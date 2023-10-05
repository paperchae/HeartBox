import wandb


def init(cfg) -> None:
    """
    Initialize wandb based on config file.

    :param cfg: project, entity, batch_size, lr, gamma, weight_decay
    :return: None
    """
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name="batch:"
        + str(cfg.train.general.batch_size)
        + "_lr:"
        + str(cfg.train.hyperparameter.lr)
        + "_gamma:"
        + str(cfg.train.hyperparameter.gamma)
        + "_weight_decay:"
        + str(cfg.train.hyperparameter.weight_decay),
    )


def log_train_valid(train_loss, valid_loss, epoch, loss_fn, log_flag) -> None:
    """
    Log train and validation loss to wandb.

    :param train_loss: last train loss
    :param valid_loss: last validation loss
    :param epoch: epoch number
    :param loss_fn: name of loss function
    :param log_flag: flag to log or not based on debug flag and wandb flag
    :return: None
    """
    if log_flag:
        wandb.log(
            {
                "train_{}_loss".format(str(loss_fn)): train_loss,
                "valid_{}_loss".format(str(loss_fn)): valid_loss,
            },
            step=epoch,
        )
    else:
        pass


def log_test(test_loss, epoch, loss_fn, log_flag) -> None:
    """
    Log test loss to wandb.

    :param test_loss: test loss when validation loss is minimum
    :param epoch: epoch number
    :param loss_fn: name of loss function
    :param log_flag: flag to log or not based on debug flag and wandb flag
    :return: None
    """
    if log_flag:
        wandb.log({"test_{}_loss".format(str(loss_fn)): test_loss}, step=epoch)
    else:
        pass
