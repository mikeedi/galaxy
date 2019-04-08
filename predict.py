from dataloader import val_transform

from shutil import copyfile
from fire import Fire
from sklearn.neighbors import NearestNeighbors

def get_image_path_by_id(idx):
    if ops.exists('data/train/data/' + idx):
        return 'data/train/data/' + idx
    elif ops.exists('data/val/data/' + idx):
        return 'data/val/data/' + idx
    else:
        raise FileNotFoundError

def similarity(encoder_predict, num):
	codes = np.load('embedding/codes.npy')
	filenames_and_coords = np.load('embedding/filenames.npy')

	nn = NearestNeighbors()
	nn.fit(codes, y=filenames)

	dist, idx = nn.kneighbors(encoder_predict, n_neighbors=num)
	
	return dist, filenames[idx]


def predict(image_path, weights, num=10, device='cpu', random_rotate=False):
	# load encoder
	encoder = Encoder(128)
	encoder.load_state_dict(torch.load(weights, map_location=device))
	encoder.eval()

	# load image
	image_name = os.path.splitext(os.path.basename(image_path))[0]
	os.mkdir(name)

	image = io.imread(image_path)
	tensor_image = val_transform(image)
	tensor_image = tensor_image.to(device)

	# find similar num galaxies
	encoder_predict = encoder(tensor_image[None])
	distances, filenames_and_coords = similarity(encoder_predict, num)



	# save txt (id, coords and probability) and images in folder with the same name as image_path but without extension
	with open(image_name + '.txt', 'w') as file:
		for fac, dist in zip(filenames_and_coords, distances):
			
			# get id and coordinate
			idx = fac[0]
			ra = fac[1]
			dec = fac[2]

			# overwrite image in new folder
			path_read = get_image_path_by_id(idx)
			id_of_predicted = os.path.splitext(os.path.basename(path_read))[0]
			path_save = os.path.join(image_name, id_of_predicted)
			copyfile(path_read, path_save)

			file.writiline(' '.join([id_of_predicted, ra, dec, dist]))



if __name__ == '__main__':
	Fire(predict)