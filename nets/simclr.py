import torch


class SimCLR(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super(SimCLR, self).__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.train.hyperparameter.in_channels
        self.out_channels = 10

        self.enconv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.in_channels, self.out_channels, kernel_size=5, stride=1
            )
        )
