import torch


class Net1D(torch.nn.Module):
    """
    1D CNN Model for detecting AFIB_OR_AFL binary classification model

    1. Doubled the depth of the model
    2. Applied GELU activation function to solve the vanishing gradient problem
    3. Applied Dropout to prevent over-fitting (0.25)
    4. By using dilated convolution, the model can learn long-term dependencies
    5. The model is designed to be able to learn the characteristics of the data by using the pooling layer
    """

    def __init__(self, cfg) -> None:
        super(Net1D, self).__init__()
        self.cfg = cfg
        self.target_length = (
            self.cfg.preprocess.data.time * self.cfg.preprocess.option.target_fs
        )
        self.num_classes = self.cfg.train.general.number_of_classes
        self.in_channels = self.cfg.train.hyperparameter.in_channels
        self.out_channels = self.cfg.train.hyperparameter.out_channels

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1
            ),
            torch.nn.BatchNorm1d(self.out_channels),
            torch.nn.Conv1d(
                self.out_channels,
                self.out_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm1d(self.out_channels * 2),
            torch.nn.GELU(),
        )
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.out_channels * 2,
                self.out_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm1d(self.out_channels * 2),
            torch.nn.Conv1d(
                self.out_channels * 2,
                self.out_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm1d(self.out_channels * 2),
            torch.nn.GELU(),
        )

        self.fc = torch.nn.Linear(self.out_channels * 2, 1)
        self.dropout = torch.nn.Dropout(self.cfg.train.hyperparameter.dropout)
        self.pool = torch.nn.MaxPool1d(self.num_classes)
        self.adaptivepool = torch.nn.AdaptiveAvgPool1d(self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.target_length)
        x = self.conv_block1(x)
        x = self.pool(self.dropout(x))
        x = self.conv_block2(x)
        x = self.pool(self.dropout(x))
        x = self.adaptivepool(x)
        x = self.fc(torch.max(x, dim=-1)[0])

        return x
