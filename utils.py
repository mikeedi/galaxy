from dataloader import mean, std
import matplotlib.pyplot as plts
from skimage import io

def show_image(img, mean=mean, std=std):
    image = img * std + mean
    image = np.clip(image, 0, 1)
    plt.tight_layout()
    io.imshow(image)


