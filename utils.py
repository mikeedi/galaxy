from dataloader import mean, std, dataloader

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd
from fire import Fire
import torch

from autoencoder import Encoder

def show_image(img, mean=mean, std=std):
    image = img * std + mean
    image = np.clip(image, 0, 1)
    plt.tight_layout()
    io.imshow(image)

def save_embeddings(weights, path='data/', device='cuda', code_size=64, batch_size=16):

    encoder = Encoder(code_size).to(device)
    encoder.load_state_dict(torch.load(weights, map_location=device))
    encoder.eval()

    train_loader, val_loader, _ = dataloader(root=path, batch_size=batch_size, 
                                             shuffle=False, transform=False, 
                                             drop_last=False)
    db = pd.read_csv('coordinates.csv')


    codes_embed = []
    filenames_embed = []
    with torch.no_grad():
        for train_batch, filenames in tqdm(train_loader):
            train_batch = train_batch.to(device)
            codes_embed.extend(encoder(train_batch).cpu().numpy())
            for i in range(len(filenames)):
                idx = filenames[i].replace('_large', '').replace('.jpg', '')
                try:
                    ra, dec = db[db['#OBJID'] == int(idx)][['RA', 'DEC']].values[0]              
                    filenames_embed.append([filenames[i], ra, dec])
                except:
                    filenames_embed.append([filenames[i], None, None])

        for val_batch, filenames in tqdm(val_loader):
            val_batch = val_batch.to(device)
            codes_embed.extend(encoder(val_batch).cpu().numpy())
            for i in range(len(filenames)):
                idx = filenames[i].replace('_large', '').replace('.jpg', '')
                try:
                    ra, dec = db[db['#OBJID'] == int(idx)][['RA', 'DEC']].values[0]              
                    filenames_embed.append([filenames[i], ra, dec])
                except:
                    filenames_embed.append([filenames[i], None, None])

    if not os.path.exists('embedding'):
        os.mkdir('embedding')
    np.save('embedding/codes.{}.npy'.format(code_size), codes_embed)
    np.save('embedding/filenames.{}.npy'.format(code_size), filenames_embed)

if __name__ == '__main__':
    Fire(save_embeddings)