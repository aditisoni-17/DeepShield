import torch
import torch.nn as nn
import torch.fft

class FrequencyBranch(nn.Module):
    def __init__(self):
        super(FrequencyBranch, self).__init__()
        self.fc = nn.Linear(224 * 224 * 3, 128)

    def forward(self, x):

        fft = torch.fft.fft2(x)
        magnitude = torch.abs(fft)

        flattened = magnitude.view(magnitude.size(0), -1)
        features = self.fc(flattened)

        return features