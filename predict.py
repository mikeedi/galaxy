from dataloader import val_transform, inference_transform_with_rotating
from autoencoder import Encoder

import os
import PIL
import numpy as np
from shutil import copyfile
from fire import Fire
from sklearn.neighbors import NearestNeighbors
from skimage import io
import torch

def round_with_None(x):
    if x is None:
        return x
    else:
        return round(x, 5)

def get_image_path_by_id(idx):
    if os.path.exists('data/train/data/' + idx):
        return 'data/train/data/' + idx
    elif os.path.exists('data/val/data/' + idx):
        return 'data/val/data/' + idx
    else:
        raise FileNotFoundError

def similarity(encoder_predict, num, code_size):

    try:
        codes = np.load('embedding/codes.{}.npy'.format(code_size))
        filenames_and_coords = np.load('embedding/filenames.{}.npy'.format(code_size))
    except:
        raise FileNotFoundError('embedding/filenames.{}.npy'.format(code_size))
    
    nn = NearestNeighbors()
    nn.fit(codes, y=filenames_and_coords)

    dist, idx = nn.kneighbors(encoder_predict, n_neighbors=num)
    return dist, filenames_and_coords[idx]


def predict(image_path, weights, code_size=64, num=10, device='cpu', random_rotate=False):
    # load encoder
    encoder = Encoder(code_size)
    encoder.load_state_dict(torch.load(weights, map_location=device))
    encoder.eval()

    # load image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if not os.path.exists(image_name):
        os.mkdir(image_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError
    image = PIL.Image.fromarray(io.imread(image_path))
    
    if random_rotate:
        best_distances = []
        best_filenames_and_coords = []
        for i in range(5):
            # find similar num galaxies
            tensor_image = inference_transform_with_rotating(image).to(device)
            with torch.no_grad():
                encoder_predict = encoder(tensor_image[None]).cpu().numpy()
            distances, filenames_and_coords = similarity(encoder_predict, num, code_size)
            distances =  distances[0]
            filenames_and_coords = filenames_and_coords[0]
            best_distances.extend(distances)
            best_filenames_and_coords.extend(filenames_and_coords)

        best_distances = np.array(best_distances)
        best_filenames_and_coords = np.array(best_filenames_and_coords)
        inds = best_distances.argsort()
        distances = best_distances[inds][:num]
        filenames_and_coords = best_filenames_and_coords[inds][:num]
    else:
        tensor_image = val_transform(image).to(device)

        # find similar num galaxies
        with torch.no_grad():
            encoder_predict = encoder(tensor_image[None]).cpu().numpy()
        distances, filenames_and_coords = similarity(encoder_predict, num, code_size)
        distances =  distances[0]
        filenames_and_coords = filenames_and_coords[0]

    # save txt (id, coords and probability) and images in folder with the same name as image_path but without extension
    with open(image_name + '/coordinates.txt', 'w') as file:
        file.write('#ID RA DEC similarity-value\n')
        for fac, dist in zip(filenames_and_coords, distances):
            
            # get id and coordinate
            idx = fac[0]
            ra = round_with_None(fac[1])
            dec = round_with_None(fac[2])
            dist = round_with_None(dist)

            # overwrite image in new folder
            path_read = get_image_path_by_id(idx)
            id_of_predicted = os.path.splitext(os.path.basename(path_read))[0]
            path_save = os.path.join(image_name, (os.path.basename(path_read)))
            copyfile(path_read, path_save)

            file.write(' '.join([str(id_of_predicted), str(ra), str(dec), str(dist)]))
            file.write('\n')


if __name__ == '__main__':
    Fire(predict)