import torch
from torch.nn.modules.loss import _Loss


class VAE_Loss(_Loss):
    
    def forward(self, z_mean, z_log_var):
        return -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).sum()
