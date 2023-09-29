from tqdm import tqdm
import wandb


def train_one_epoch(**kwargs) -> float:
    """
    Train model for one epoch
    # TODO: Implement training function which trains model for one epoch and return average training loss in one epoch.
        1. Set model to train mode
        2. Iterate over data loader
        3. Compute loss
        4. Backpropagate loss
        5. Update parameters

    Pseudo code:
        for batch in data_loader:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    """
    total_cost = 0.0
    kwargs['model'].train() # set model to train mode (dropout, batchnorm, etc.)
    with tqdm(kwargs['loader'], desc=f"Train Epoch {kwargs['epoch']}") as batch_iterator:
        for x, y, _ in batch_iterator:
            kwargs['optimizer'].zero_grad()

            x = x.to(kwargs['device'])
            y = y.to(kwargs['device'])

            output = kwargs['model'](x)
            loss = kwargs['loss'](output.view(-1), y)
            total_cost += loss.item()
            loss.backward()
            kwargs['optimizer'].step()

            batch_iterator.set_postfix(loss=total_cost / (batch_iterator.n + 1))

    kwargs['scheduler'].step()

    return total_cost / len(batch_iterator)
