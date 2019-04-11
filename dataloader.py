import os

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms, utils, datasets
import numpy as np


class ImageFolderWithGalaxyId(datasets.ImageFolder):
    """Custom dataset that includes galaxy id. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithGalaxyId, self).__getitem__(index)
        tensor_tuple, _ = original_tuple
        # the image file path
        path = self.imgs[index][0]
        # galaxy id = filename without extension
        filename = os.path.basename(path)
        # make a new tuple that includes original and the path
        tuple_with_path = (tensor_tuple, filename)
        return tuple_with_path


# mean and std of all dataset
mean = [0.0804, 0.0667, 0.0513]
std = [0.1367, 0.1182, 0.0818]

# all transformation
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomAffine(0, translate=(0, 0.1)),
        transforms.Resize(256),        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

val_transform = transforms.Compose([
        transforms.Resize(256),        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

inference_transform_with_rotating = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.Resize(256),        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

def dataloader(root='data/', batch_size=32, shuffle=True, transform=True, drop_last=True):

    if not os.path.exists(root):
        raise FileNotFoundError("image files was't found: ", root)

    if transform:
        train_dataset = ImageFolderWithGalaxyId(root=root + 'train/',
                                         transform=train_transform
                                        )
    else:
        train_dataset = ImageFolderWithGalaxyId(root=root + 'train/',
                                         transform=val_transform
                                        )
    
    val_dataset = ImageFolderWithGalaxyId(root=root + 'val/',
                                         transform=val_transform
                                        )

    # create train evaluator
    indices = np.arange(len(train_dataset))
    random_indices = np.random.permutation(indices)[:len(val_dataset)]
    train_subset = Subset(train_dataset, indices=random_indices)

    # dataset for training
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             shuffle=shuffle
                            )
    
    # dataset for evaluation
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last
                            )
    
    # dataset from training dataset for evaluation with the same size of val_loader
    train_eval_loader = DataLoader(train_subset, batch_size=batch_size, 
                             shuffle=shuffle, drop_last=drop_last
                            )
    return train_loader, val_loader, train_eval_loader


if __name__ == '__main__':
    _, val_loader, _ = dataloader(batch_size=1)

    for tensor, filename in val_loader:
        print(tensor)
        print(filename)
        break