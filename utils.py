from dataloader import mean, std
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd
from fire import Fire

from autoencoder import Encoder

def show_image(img, mean=mean, std=std):
    image = img * std + mean
    image = np.clip(image, 0, 1)
    plt.tight_layout()
    io.imshow(image)

def save_embeddings(weights, path='data/', device='cpu'):

	encoder = Encoder(128)
	encoder.load_state_dict(torch.load(weights, map_location=device))
	encoder.eval()

	train_loader, val_loader, _ = dataloader(root=path, batch_size=16, 
	                                         shuffle=False, transform=False, 
	                                         drop_last=False)
	db = pd.read_csv('GalaxyZoo1_DR_table2.csv')


	codes = []
	filenames = []
	with torch.no_grad():
	    for train_batch, filename in tqdm(train_loader):
	    	train_batch = train_batch.to(device)
	        codes.extend(encoder(train_batch).cpu().numpy())
	        for i in range(len(filename)):
	        	idx = filenames[i]
				ra, dec = db[db['#OBJID'] == idx][['RA', 'DEC']].values[0]              
	        	filenames.append([idx, ra, dec])
	        
	    for val_batch, filename in tqdm(val_loader):
	    	val_batch = val_batch.to(device)
	        codes.extend(encoder(val_batch).cpu().numpy())
	        for i in range(len(filename)):
	        	idx = filenames[i]
				ra, dec = db[db['#OBJID'] == idx][['RA', 'DEC']].values[0]              
	        	filenames.append([idx, ra, dec])

	np.save('embedding/codes.npy', codes)
	np.save('embedding/filenames.npy', filenames)

if __name__ == '__main__':
	Fire(save_embeddings)