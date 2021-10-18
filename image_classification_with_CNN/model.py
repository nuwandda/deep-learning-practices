import os
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from matplotlib import pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


class Model(nn.Module):
    def __init__(self, device):
        # Inherits the base class features
        super(Model, self).__init__()

        # Create the model layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ).to(device)

        self.classifier = nn.Sequential(
            # Flattens a contiguous range of dims into a tensor
            nn.Flatten(),
            # During training, randomly zeroes some of the elements of the input tensor with probability p
            # using samples from a Bernoulli distribution. Each channel will be zeroed out independently
            # on every forward call.
            # An effective technique for regularization and preventing the co-adaptation of neurons
            nn.Dropout(0.25),
            # Applies a linear transformation to the incoming data
            nn.Linear(4096, 256),
            # Applies the rectified linear unit function element-wise. Using the function max(0, x)
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256, 10)
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)

        return x
