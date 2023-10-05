import torch
import wandb
from tqdm import tqdm


def validation_one_epoch(**kwargs) -> float:
    kwargs["model"].eval()
    with tqdm(
        kwargs["loader"], desc=f"Valid Epoch {kwargs['epoch']}"
    ) as batch_iterator:
        total_cost = 0.0
        with torch.no_grad():
            for x, y, _ in batch_iterator:
                x = x.to(kwargs["device"])
                y = y.to(kwargs["device"])

                output = kwargs["model"](x)
                loss = kwargs["loss"](output.view(-1), y)
                total_cost += loss.item()

                batch_iterator.set_postfix(loss=total_cost / (batch_iterator.n + 1))
            # if kwargs['wandb']:
            #     wandb.log({"valid_loss": total_cost / len(batch_iterator)})
            return total_cost / len(batch_iterator)
