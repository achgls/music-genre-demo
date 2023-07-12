from typing import Union, List, Tuple

from torch import Tensor
import torch
import torch.nn as nn

from .se_module import SELayer


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop = nn.Dropout(dropout) if dropout is not None else None
        self.se_layer = SELayer(channel=out_channels, reduction=reduction) if reduction is not None else None
        self.relu = nn.ReLU()  # Not specified in the paper, but would seem natural

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.se_layer is not None:
            x = self.se_layer(x)
        x = self.relu(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class CNN(nn.Module):
    def __init__(
            self,
            num_classes: int,
            in_channels: int = 1,
            filters: Union[int, List[int], Tuple[int]] = (32, 64, 128, 256, 128),
            reduction: Union[int, List[int], Tuple[int]] = 32,
            dropout: Union[float, List[float], Tuple[float]] = .25,
            fc_dropout: float = None,
            fc_bias: bool = False,
            pool: str = "flatten",
            input_size: Tuple[int] = (513, 130)
    ):
        """
        Standard CNN architecture as used in the paper used as a guideline for the project
        :param num_classes: number of output classes for classification
        :param in_channels: number of channels in the input, default=1 for STFT/spec
        :param filters: sequence of integers corresponding to the number of convolutions kernels for each conv layer
        :param reduction: integer or sequence of integers corresponding to the SE reduction ratio for each layer
        :param dropout: dropout rate to apply at the end of each convolutional layer, single value or sequence
        :param fc_dropout: if not None, dropout rate to apply on top of the final "feature" vector, default = None
        :param fc_bias: boolean indicating whether bias should be used in the final, classification layer, default=False
        :param pool: 'flatten' or 'GAP'
        :param input_size: expected input size (HxW), used if pool = "flatten" to init. FC layer with correct # of units
        """
        super().__init__()

        n_layers = len(filters)

        # ----- Sanity checks -----
        if reduction is None or isinstance(reduction, int):
            self.reduction = [reduction] * n_layers
        else:
            assert len(reduction) == n_layers
            self.reduction = reduction
        if dropout is None or isinstance(dropout, float):
            assert 0. <= dropout <= 1, "Dropout ratio must be a floating point value in the range [0; 1]"
            self.dropout = [dropout] * n_layers
        else:
            assert len(dropout) == n_layers
            self.dropout = dropout

        # ----- Make convolutional layers -----
        self.convolutional_core = nn.Sequential()
        self.convolutional_core.add_module(
            "ConvBlock0" + ("_SE" if self.reduction[0] else ''),
            ConvBlock(
                in_channels=in_channels,
                out_channels=filters[0],
                reduction=self.reduction[0],
                dropout=self.dropout[0]
            ))
        for i in range(1, n_layers):
            self.convolutional_core.add_module(
                "ConvBlock%d" % i + ("_SE" if self.reduction[i] else ''),
                ConvBlock(
                    in_channels=filters[i-1],
                    out_channels=filters[i],
                    reduction=self.reduction[i],
                    dropout=self.dropout[i]
                ))

        # ----- Fully-connected layer with pooling -----
        if pool == "GAP":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(filters[-1], num_classes, bias=fc_bias)
        elif pool == "flatten":
            assert input_size is not None, "When 'flatten' is applied, input size must be specified and fixed."
            self.pool = nn.Flatten()
            self.fc = nn.Linear(
                in_features=(input_size[0] // (2 ** n_layers)) * (input_size[1] // (2 ** n_layers)) * filters[-1],
                out_features=num_classes, bias=fc_bias)
        else:
            raise AssertionError("Final pooling for fully-connected layer should be either 'flatten' or 'GAP'.")

        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout is not None else None

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.convolutional_core(x)
        feature_vector = self.pool(x)
        feature_vector = torch.flatten(feature_vector, 1)
        if self.fc_dropout is not None:
            feature_vector = self.fc_dropout(feature_vector)
        predictions = self.fc(feature_vector)
        return predictions

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
