import os

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms, utils, datasets

import numpy as np

# mean and standart deviation of all dataset
mean = [0.0804, 0.0667, 0.0513]
std = [0.1367, 0.1182, 0.0818]
image_size = 264

# all transformation
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0, 0, 0, 0.1),
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

def dataloader(root='data/', batch_size=32, shuffle=True):

    if not os.path.exists(root):
        raise FileNotFoundError("image files was't found: ", root)

    train_dataset = datasets.ImageFolder(root=root + 'train/',
                                         transform=train_transform
                                        )
    val_dataset = datasets.ImageFolder(root=root + 'val/',
                                         transform=val_transform
                                        )

    # create train evaluator
    indices = np.arange(len(train_dataset))
    random_indices = np.random.permutation(indices)[:len(val_dataset)]
    train_subset = Subset(train_dataset, indices=random_indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             shuffle=shuffle
                            )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=True
                            )
    train_eval_loader = DataLoader(train_subset, batch_size=batch_size, 
                             shuffle=shuffle, drop_last=True
                            )
    return train_loader, val_loader, train_eval_loader