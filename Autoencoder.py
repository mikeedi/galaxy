import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    """Encoder implementation"""
    def __init__(self, code_size):
        super(Encoder, self).__init__()
        self.code_size = code_size
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # self.dense_1 = nn.Sequential(
        #     nn.Linear(16*8*8, 16*8*4),
        #     nn.BatchNorm1d(16*8*4),
        #     nn.ELU()
        # )
        self.dense_2 = nn.Sequential(
            nn.Linear(128*8*8, self.code_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = x.view(x.size(0), -1)
        # x = self.dense_1(x)
        x = self.dense_2(x)
        return x

class Decoder(nn.Module):
    """Decoder implementation"""
    def __init__(self, code_size):
        super(Decoder, self).__init__()
        self.code_size = code_size
        # self.dense_1 = nn.Sequential( 
        #     nn.Linear(self.code_size, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ELU()
        # )
        self.dense_2 = nn.Sequential(
            nn.Linear(self.code_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU()
        )
        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.deconv_block_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.deconv_block_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.deconv_block_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        self.deconv_block_5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        self.deconv_out = nn.Sequential(
            nn.ConvTranspose2d(256, 3, kernel_size=(3, 3), padding=1, stride=1),
        )

    def forward(self, x):
        # x = self.dense_1(x)
        x = self.dense_2(x)
        x = x.view(x.size(0), 16, 8, 8)
        x = self.deconv_block_1(x)
        x = nn.functional.pad(x, (1, 0, 1, 0))
        x = self.deconv_block_2(x)
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.deconv_block_3(x)
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.deconv_block_4(x)
        x = nn.functional.pad(x, (1, 0, 1, 0))
        x = self.deconv_block_5(x)
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.deconv_out(x)
        return x

class Autoencoder(nn.Module):
    """Autoencoder implementation"""
    def __init__(self, code_size):
        super(Autoencoder, self).__init__()
        self.code_size = code_size
        self.encoder = Encoder(code_size)
        self.decoder = Decoder(code_size)

    def forward(self, x):
        code = self.encoder(x)
        decode = self.decoder(code)
        return decode