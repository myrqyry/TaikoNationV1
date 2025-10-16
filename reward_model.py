import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    A simple neural network to predict the quality of a Taiko chart.
    This model is a placeholder and is not trained on actual human feedback.
    It takes the mean of the transformer's hidden states as input.
    """
    def __init__(self, input_size, hidden_size=128):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of aggregated chart features.
                              Shape: [batch_size, input_size]
        Returns:
            torch.Tensor: A scalar reward score for each chart in the batch.
                          Shape: [batch_size, 1]
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out