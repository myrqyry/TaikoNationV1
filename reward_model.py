import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    A simple MLP to predict a scalar reward score for a generated chart.
    It is designed to take a concatenated vector of features as input.
    """
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1):
        """
        Args:
            input_size (int): The size of the input feature vector. This will be a
                              combination of model hidden states, quantitative metrics, etc.
            hidden_size (int): The size of the hidden layer.
            dropout_rate (float): The dropout rate to apply for regularization.
        """
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input feature vector.

        Returns:
            torch.Tensor: A scalar reward value.
        """
        return self.model(x)