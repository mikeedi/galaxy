from dataloader import mean, std

def show_image(img, mean=mean, std=std):
    image = img * std + mean
    image = np.clip(image, 0, 1)
    plt.tight_layout()
    io.imshow(image)


