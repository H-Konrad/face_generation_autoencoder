import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class celeba_encoder_version_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding = 1), # 64 -> 64
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 64 -> 32
        nn.Conv2d(8, 16, 3, padding = 1), # 32 -> 32
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 32 -> 16
        nn.Conv2d(16, 32, 3, padding = 1), # 16 -> 16
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16 -> 8
        nn.Conv2d(32, 64, 3, padding = 1), # 8 -> 8
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 8 -> 4
        nn.Conv2d(64, 128, 3, padding = 1), # 4 -> 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 4 -> 2
        nn.Conv2d(128, 256, 3, padding = 1), # 2 -> 2
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 2 -> 1
        nn.Flatten(),
        nn.Linear(1 * 1 * 256, 128),
        nn.ReLU(),
        nn.Linear(128, 40),
        nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return x

class celeba_encoder_version_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, 5, padding = 2, stride = 2), # 128 -> 64
        nn.ReLU(),
        nn.Conv2d(8, 16, 5, padding = 2, stride = 2), # 64 -> 32
        nn.ReLU(),
        nn.Conv2d(16, 32, 5, padding = 2, stride = 2), # 32 -> 16
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, padding = 2, stride = 2), # 16 -> 8
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding = 1, stride = 2), # 8 -> 4
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding = 1, stride = 2), # 4 -> 2
        nn.ReLU(),
        nn.Conv2d(256, 512, 3, padding = 1, stride = 2), # 2 -> 1
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1 * 1 * 512, 256),
        nn.ReLU(),
        nn.Linear(256, 40),
        nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return x

class celeba_autoencoder_version_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding = 1), # 64 -> 64
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 64 -> 32
        nn.Conv2d(8, 16, 3, padding = 1), # 32 -> 32
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 32 -> 16
        nn.Conv2d(16, 32, 3, padding = 1), # 16 -> 16
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16 -> 8
        nn.Conv2d(32, 64, 3, padding = 1), # 8 -> 8
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 8 -> 4
        nn.Conv2d(64, 128, 3, padding = 1), # 4 -> 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 4 -> 2
        nn.Conv2d(128, 256, 3, padding = 1), # 2 -> 2
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 2 -> 1
        nn.Flatten(),
        nn.Linear(1 * 1 * 256, 128),
        nn.ReLU(),
        nn.Linear(128, 40),
        nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 1 * 1 * 256),
        nn.ReLU(),
        nn.Unflatten(1, (256, 1, 1)),
        nn.ConvTranspose2d(256, 128, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, 4, padding = 1, stride = 2),
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

class celeba_autoencoder_version_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride = 2), # 64 -> 31
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride = 2), # 31 -> 15
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride = 2), # 14 -> 7
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride = 2), # 7 -> 3
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride = 2), # 3 -> 1
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1 * 1 * 256, 170),
        nn.ReLU(),
        nn.Linear(170, 84),
        )

        self.decoder = nn.Sequential(
        nn.Linear(84, 199),
        nn.ReLU(),
        nn.Linear(199, 398),
        nn.ReLU(),
        nn.Linear(398, 512),
        nn.ReLU(),
        nn.Unflatten(1, (512, 1, 1)),
        nn.ConvTranspose2d(512, 256, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 4, padding = 1, stride = 2),
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class celeba_autoencoder_version_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 8, 5, padding = 2, stride = 2), # 128 -> 64
        nn.ReLU(),
        nn.Conv2d(8, 16, 5, padding = 2, stride = 2), # 64 -> 32
        nn.ReLU(),
        nn.Conv2d(16, 32, 5, padding = 2, stride = 2), # 32 -> 16
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, padding = 2, stride = 2), # 16 -> 8
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding = 1, stride = 2), # 8 -> 4
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding = 1, stride = 2), # 4 -> 2
        nn.ReLU(),
        nn.Conv2d(256, 512, 3, padding = 1, stride = 2), # 2 -> 1
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1 * 1 * 512, 256),
        nn.ReLU(),
        nn.Linear(256, 40),
        nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
        nn.Linear(40, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 1 * 1 * 512),
        nn.LeakyReLU(),
        nn.Unflatten(1, (512, 1, 1)),
        nn.ConvTranspose2d(512, 256, 4, padding = 1, stride = 2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(256, 128, 4, padding = 1, stride = 2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 6, padding = 2, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 6, padding = 2, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 6, padding = 2, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, 6, padding = 2, stride = 2),
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

class celeba_autoencoder_version_4(nn.Module): 
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding = 1, stride = 2), # 128 -> 64
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, 3, padding = 1, stride = 2), # 64 -> 32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, 3, padding = 1, stride = 2), # 32 -> 16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, 3, padding = 1, stride = 2), # 16 -> 8
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128, 256, 3, padding = 1, stride = 2), # 8 -> 4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(256, 512, 3, padding = 1, stride = 2), # 4 -> 2
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(2 * 2 * 512, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        )

        self.decoder = nn.Sequential(
        nn.Linear(128, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Linear(512, 2 * 2 * 512),
        nn.BatchNorm1d(2 * 2 * 512),
        nn.LeakyReLU(),
        nn.Unflatten(1, (512, 2, 2)),
        nn.ConvTranspose2d(512, 256, 4, padding = 1, stride = 2), # 2 -> 4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(256, 128, 4, padding = 1, stride = 2), # 4 -> 8
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2), # 8 -> 16
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64, 32, 4, padding = 1, stride = 2), # 16 -> 32
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(32, 16, 4, padding = 1, stride = 2), # 32 -> 64
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(16, 3, 4, padding = 1, stride = 2), # 64 -> 128
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x