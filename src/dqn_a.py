import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    params:
        input_shape: eg. (4, 84, 84)
        n_actions: eg. 6 for Pong
        conv_layers: list of conv layer configs
        fc_units: hidden size for fully connected layer
        activation: the activation function
    """
    def __init__(
        self,
        input_shape,
        n_actions,
        conv_layers=None,
        fc_units=512,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.activation = activation

        self.conv = self._build_conv_layers(conv_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, fc_units),
            activation(),
            nn.Linear(fc_units, n_actions)
        )

    def _build_conv_layers(self, conv_layers_config):
        layers = []
        in_channels = self.input_shape[0]
        for cfg in conv_layers_config:
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=cfg["out_channels"],
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"]
            ))
            layers.append(self.activation())
            in_channels = cfg["out_channels"]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
