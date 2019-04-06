import torch
from torch.nn.modules.loss import _Loss


class VAE_Loss(_Loss):

    def __init__(self, reduction='mean'):
        super(VAE_Loss, self).__init__(reduction=reduction)


    def forward(self, z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), axis=-1)
