import argparse

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from Autoencoder import Autoencoder
from dataloader import dataloader
from utils import show_image

def train(num_epochs=30, batch_size=64, use_cuda=True, code_size=64,
                train_size=0.75, steps=2000, save_images=True):

    if train_size > 1.:
        raise ValueError("train_size could not be more 1")
    
    model = Autoencoder(code_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_loss_save = []
    test_loss_save = []

    if use_cuda:
        model = model.cuda()

    train_loader, test_loader = dataloader(root='data/', batch_size=batch_size, train_size=train_size)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
    
            running_loss = 0.0
    
            # get the inputs
            inputs, _ = data
            if use_cuda:
                inputs = inputs.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % (steps - 1) == 1:  # print every steps mini-batches
                print('[%d, %d] training loss: %.8f' %
                          (epoch + 1, i + 1, running_loss/steps))
        train_loss_save.append(running_loss / len(inputs))

        # if there is not test_data skip validation phase
        if train_size == 1.:
            continue

        # validation step
        for i, data in enumerate(test_loader, 0):
            # get the inputs
            inputs, _ = data
            if use_cuda:
                inputs = inputs.cuda()

            # forward only
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # print statistics
            running_loss += loss.item()
        test_loss_save.append(running_loss/len(test_loader))
        print('validation loss: %.8f' % (train_loss_save[-1]))
        if save_images:
            for i in np.random.random_integers(0, len(test_loader), 10):
                image, _ = test_loader[i]
                plt.subplot(1, 2, 1)
                show_image(np.swapaxes(image.numpy(), 0, 2))
                predict_image = model(image[None])[0].detach()
                plt.subplot(1, 2, 2)
                show_image(np.swapaxes(predict_image.numpy(), 0, 2))
                plt.savefig('images/{}_{}'.format(epoch, i))


    print('Finished Training')
    print('Best train epoch: ', np.argmax(train_loss_save), np.max(train_loss_save))
    print('Best test epoch: ', np.argmax(test_loss_save), np.max(test_loss_save))
    torch.save(model, 'Autoencoder.pt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convolution autoencoder')

    parser.add_argument('--num-epochs', type=int, default=30, metavar='NE',
                        help='num iterations (default: 30)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--code-size', type=int, default=64, metavar='CS',
                        help='code size (default: 64)')
    parser.add_argument('--train-size', type=float, default=0.75, metavar='TS',
                        help='train size (default: 0.75); between 0 to 1; if 1 then without validating')
    parser.add_argument('--steps', type=int, default=2000, metavar='ST',
                        help='print statistics every steps (default: 2000)')
    parser.add_argument('--save-images', type=bool, default=True, metavar='CUDA',
                        help='save images after and before autoencoder (default: True)')

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size 
    use_cuda = args.use_cuda
    code_size = args.code_size
    train_size = args.train_size 
    steps = args.steps
    save_images = args.save_images
    train(num_epochs, batch_size, use_cuda, code_size,
                    train_size, steps, save_images)