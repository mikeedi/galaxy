import os

import torch
from torch import nn
from dataloader import dataloader, train_transform
from vae_loss import VAE_Loss
from vae import Encoder, Decoder

latent_size = 128
device = 'cuda:0'
root = 'data/'
start_lr = 0.001
batch_size = 32
log_interval = 50
if not os.path.exists('images_vae/'):
    os.mkdirs('images_vae/')

train_loader, val_loader, _ = dataloader(root=root, 
                                            batch_size=batch_size)


reconstruct_criterion = nn.MSELoss().to(device)
kl_divergence = VAE_Loss().to(device)
encoder = Encoder(latent_size).to(device)
decoder = Decoder(latent_size).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=start_lr)

num_epoch = 100

for epoch in range(num_epoch):

    encoder.train()
    decoder.train()
    for i, (img, _) in enumerate(train_loader):
    
        optimizer.zero_grad()       
        img = img.to(device)

        # predict distribution
        z_mean, z_log_var = encoder(img)

        # sample z
        eps = torch.randn((batch_size, latent_size)).to(device)
        z_sample = z_mean + eps * torch.exp(z_log_var / 2)

        # reconstruct image
        decoded = decoder(z_sample)

        divergense_loss = kl_divergence(z_mean, z_log_var)
        reconstruct_loss = reconstruct_criterion(img, decoded)

        loss = divergense_loss + reconstruct_loss
        loss.backward()
        optimizer.step()

        if iteration % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                  .format(epoch, i, len(train_loader), loss))
        
    with torch.no_grad():    
        encoder.eval()
        decoder.eval()
        for i, (img, _) in enumerate(val_loader):
                img = img.to(device)

                # predict distribution
                z_mean, z_log_var = encoder(img)

                # sample z
                eps = torch.randn((batch_size, latent_size)).to(device)
                z_sample = z_mean + eps * torch.exp(z_log_var / 2)

                # reconstruct image
                decoded = decoder(z_sample)

                divergense_loss = kl_divergence(z_mean, z_log_var)
                reconstruct_loss = reconstruct_criterion(img, decoded)

                loss = divergense_loss + reconstruct_loss

                if iteration % log_interval == 0:
                    print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                          .format(epoch, i, len(train_loader), loss))
        
        np.random.seed(46)
        for i in np.random.random_integers(0, len(val_loader), 10):
            image, _ = val_loader.dataset[i]
            image = image.to(device)
            predict_image = model(image[None])[0].detach().cpu()
            plt.subplot(1, 2, 1)
            show_image(np.swapaxes(image.cpu().numpy(), 0, 2))
            plt.subplot(1, 2, 2)
            show_image(np.swapaxes(predict_image.numpy(), 0, 2))
            plt.savefig('images_vae/{}_{}'.format(epoch, i))        

