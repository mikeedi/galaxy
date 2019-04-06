import torch
from torch import nn
from dataloader import dataloader, train_transform
from vae_loss import VAE_Loss
from vae import Encoder, Decoder

latent_size = 128
device = 'cuda:1'
root = 'data/'
start_lr = 0.001


train_loader, val_loader, _ = dataloader(root=root, 
                                            batch_size=batch_size)


reconstruct_criterion = nn.MSELoss().to(device)
kl_divergence = VAE_Loss().to(device)
encoder = Encoder(latent_size).to(device)
decoder = Decoder(latent_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

num_epoch = 100

for epoch in range(num_epoch):
	for i, (img, _) in train_loader:
	
		optimizer.zero_grad()		
		img = img.to(device)

		# predict distribution
		z_mean, z_log_var = encoder(img)

		# sample z
		eps = torch.randn((batch_size))
		z_sample = z_mean + eps * torch.exp(z_log_var / 2)

		# reconstruct image
		decoded = decoder(z_sample)

		divergense_loss = kl_divergence(z_mean, z_log_var)
		reconstruct_loss = reconstruct_criterion(img, decoded)

		loss = divergense_loss + reconstruct_loss
		loss.backward()
		optimizer.step()

		print(loss.cpu().item())
		print(divergense_loss.cpu().item())
		print(reconstruct_loss.cpu().item())
		