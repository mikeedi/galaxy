import os

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, datasets


# mean and standart deviation of all dataset
mean = [0.0804, 0.0667, 0.0513]
std = [0.1367, 0.1182, 0.0818]
image_size = 264

# all transformation
data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(),
        transforms.Resize(256),        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

def dataloader(root='data/', batch_size=32, shuffle=True, train_size=0.8):

    if not os.path.exists(root):
        raise FileNotFoundError("image files was't found")

    dataset = datasets.ImageFolder(root=root,
                              transform=data_transform)

    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             shuffle=shuffle
                            )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=shuffle
                            )

    return train_loader, test_loader


