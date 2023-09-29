import torch
from tqdm import tqdm


# https://bcho.tistory.com/1206
# https://data-newbie.tistory.com/155
# https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
# https://yuevelyne.tistory.com/10

def inference(**kwargs):  # -> Dict[float, bool]: index:[prob, pred]
    # TODO: change return value to Dict[float, bool]
    """
    1. Inference External Data
    2. Save inference result in data/index/external/external_test_index.csv using index
    3. Plot histogram of inference data by age (compare with Internal train/test data)

    :param kwargs: model, external loader, device(gpu)
    :return: Inference Probability, Prediction, Index of External Data
    """
    kwargs['model'].eval()  # == model.train(False)
    AFIB_OR_AFL_PROB = torch.empty(1)
    AFIB_OR_AFL = torch.empty(1)
    index = torch.empty(1)
    with tqdm(kwargs['loader'], desc='Testing External Data') as batch_iterator:
        with torch.no_grad():
            for x, _, idx in batch_iterator:
                x = x.to(kwargs['device'])
                idx = idx

                # model output
                output = torch.sigmoid(kwargs['model'](x).view(-1))
                prediction = ((output.view(-1)) > 0.5).to(torch.long)

                # Concatenate all inference result
                AFIB_OR_AFL_PROB = torch.cat((AFIB_OR_AFL_PROB, output.detach().cpu()), dim=0)
                AFIB_OR_AFL = torch.cat((AFIB_OR_AFL, prediction.detach().cpu()), dim=0)
                index = torch.cat((index, idx), dim=0)

    return AFIB_OR_AFL_PROB[1:], AFIB_OR_AFL[1:], index[1:]


def test_one_epoch(compute_metric: bool = False, **kwargs):
    """
    Test model using Internal Test Data
    If compute_metric is True: return AFIB_OR_AFL_PROB, AFIB_OR_AFL, label, index for computing metrics
    Else: return average test loss in one epoch

    :param compute_metric: If True, Compute metrics
    :param kwargs:
    :return: Average test loss in epoch or AFIB_OR_AFL_PROB, AFIB_OR_AFL, label, index
    """
    kwargs['model'].eval()
    if compute_metric:
        AFIB_OR_AFL_PROB = torch.empty(1)
        AFIB_OR_AFL = torch.empty(1)
        label = torch.empty(1)
        index = torch.empty(1)
    with tqdm(kwargs['loader'], desc=f'Test Epoch {kwargs["epoch"]}') as batch_iterator:
        total_cost = 0.0
        with torch.no_grad():
            for x, y, idx in batch_iterator:
                # TODO: Check if x, y is already on device + check train, valid function also.
                x = x.to(kwargs['device'])  # isn't it already on device?
                y = y.to(kwargs['device'])

                output = kwargs['model'](x)
                loss = kwargs['loss'](output.view(-1), y)
                total_cost += loss.item()
                if compute_metric:
                    output = torch.sigmoid(output.view(-1))
                    # prediction = ((output.view(-1)) > 0.5).to(torch.long)
                    prediction = ((output.view(-1)) > kwargs['threshold']).to(torch.long)
                    AFIB_OR_AFL_PROB = torch.cat((AFIB_OR_AFL_PROB, output.detach().cpu()), dim=0)
                    AFIB_OR_AFL = torch.cat((AFIB_OR_AFL, prediction.detach().cpu()), dim=0)
                    label = torch.cat((label, y.detach().cpu()), dim=0)
                    index = torch.cat((index, idx), dim=0)

                batch_iterator.set_postfix(loss=total_cost / (batch_iterator.n + 1))

            if compute_metric:
                return AFIB_OR_AFL_PROB[1:], AFIB_OR_AFL[1:], label[1:], index[1:]
            else:
                return total_cost / len(batch_iterator)
