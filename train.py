import argparse
import os

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torch.optim import Adam, Adadelta, Adagrad, SparseAdam, \
                        Adamax, SGD, RMSprop
import torch.nn as nn
import torch

from ignite.engine import Engine, _prepare_batch, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

from Autoencoder import Autoencoder
from dataloader import dataloader, train_transform
from utils import show_image


def train(root='data/', num_epochs=30, batch_size=64, device='cuda', 
        code_size=64, log_interval=20, save_images=True, 
        name_optim='Adam', start_lr=0.001, gamma=0.9):    

    model = Autoencoder(code_size).to(device)
    criterion = nn.MSELoss()
    optimizers = {
        'Adam': Adam,
        'Adadelta': Adadelta,
        'Adagrad': Adagrad,
        'SparseAdam': SparseAdam,
        'Adamax': Adamax,
        'SGD': SGD,
        'RMSprop': RMSprop,
    }

    if name_optim not in optimizers:
        raise ValueError('There is no such optimizer: {}'.format(name_optim))

    optimizer = optimizers[name_optim](model.parameters(), lr=start_lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

    train_loader, val_loader, train_eval_loader = dataloader(root=root, 
                                                batch_size=batch_size)

    print('Train-size: ', len(train_loader.dataset))
    print('Test-size: ', len(val_loader.dataset))
    print('Train-eval-size: ', len(train_eval_loader.dataset))

    def create_unsupervised_evaluator(model, metrics={}, device=None):
        if device:
            model.to(device)

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                x, _ = _prepare_batch(batch, device=device)
                x_pred = model(x)
                return x_pred, x

        engine = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(engine, name)

        return engine

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, _ = _prepare_batch(batch, device=device)
        x_pred = model(x)
        loss = criterion(x_pred, x)
        loss.backward()
        optimizer.step()
        return loss.item() / len(batch)

    trainer = Engine(process_function)

    metrics = {
    'avg_loss': Loss(criterion)
    }
    train_evaluator = create_unsupervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_unsupervised_evaluator(model, metrics=metrics, device=device)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                  .format(engine.state.epoch, 
                             iteration, 
                             len(train_loader), 
                             engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_offline_train_metrics(engine):
        epoch = engine.state.epoch
        print("Compute train metrics...")
        metrics = train_evaluator.run(train_eval_loader).metrics
        print("Training Results - Epoch: {}  Average Loss: {:.4f}"
              .format(engine.state.epoch, 
                          metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        epoch = engine.state.epoch
        print("Compute validation metrics...")
        metrics = val_evaluator.run(val_loader).metrics
        print("Validation Results - Epoch: {}  Average Loss: {:.4f}"
              .format(engine.state.epoch, 
                          metrics['avg_loss']))

    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_scheduler(engine):
        lr_scheduler.step()
        # Вывод значений скорости обучения:
        if len(optimizer.param_groups) == 1:
            lr = float(optimizer.param_groups[0]['lr'])
            print("Learning rate: {}".format(lr))
        else:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = float(param_group['lr'])
                print("Learning rate (group {}): {}".format(i, lr)) 

    def score_function(engine):
        val_avg_accuracy = engine.state.metrics['avg_loss']
        return val_avg_accuracy

    best_model_saver = ModelCheckpoint("best_models",  
                                       filename_prefix="model",
                                       score_name="val_accuracy",  
                                       score_function=score_function,
                                       n_saved=3,
                                       save_as_state_dict=True,
                                       create_dir=True)
    val_evaluator.add_event_handler(Events.COMPLETED, 
                                    best_model_saver, 
                                    {"best_model": model})

    training_saver = ModelCheckpoint("checkpoint",
                                     filename_prefix="checkpoint",
                                     save_interval=1000,
                                     n_saved=1,
                                     save_as_state_dict=True,
                                     create_dir=True)
    to_save = {
        "model": model, 
        "optimizer": optimizer, 
        "lr_scheduler": lr_scheduler, 
        'code_size': code_size,
        'batch_size': batch_size,
        'optimizer': name_optim,
        'start_learning_rate': start_lr,
        'gamma': gamma,
        'train_transform': train_transform,
    } 
    trainer.add_event_handler(Events.ITERATION_COMPLETED, training_saver, to_save)

    early_stopping = EarlyStopping(patience=10,     
                                  score_function=score_function, 
                                  trainer=trainer)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)

    if save_images:
        if not os.path.exists('images/'):
            os.mkdir('images/')
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_before_and_after(engine):
            model.eval()
            epoch = engine.state.epoch
            np.random.seed(1)
            print('Test model on 10 images...')
            for i in np.random.random_integers(0, len(val_loader), 10):
                image, _ = val_loader.dataset[i]
                image = image.to(device)
                with torch.no_grad():
                    predict_image = model(image[None])[0].detach().cpu()
                plt.subplot(1, 2, 1)
                show_image(np.swapaxes(image.cpu().numpy(), 0, 2))
                plt.subplot(1, 2, 2)
                show_image(np.swapaxes(predict_image.numpy(), 0, 2))
                plt.savefig('images/{}_{}'.format(epoch, i))        

    trainer.run(train_loader, num_epochs)
    print('THE-END')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convolution autoencoder')

    parser.add_argument('--root', type=str, default='data/', metavar='R',
                        help='folder with folder with images (default: data/)')
    parser.add_argument('--num-epochs', type=int, default=30, metavar='NE',
                        help='num iterations (default: 30)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda', metavar='CUDA',
                        help='use cuda or cpu (default: cuda)')
    parser.add_argument('--code-size', type=int, default=64, metavar='CS',
                        help='code size (default: 64)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='LI',
                        help='print statistics every log-interval (default: 20)')
    parser.add_argument('--save-images', type=bool, default=True, metavar='CUDA',
                        help='save images after and before autoencoder (default: True)')

    parser.register('type', bool, (lambda x: x.lower() in ("yes", "true", "t", "1")))
    args = parser.parse_args()

    root = args.root
    num_epochs = args.num_epochs
    batch_size = args.batch_size 
    device = args.device
    code_size = args.code_size
    log_interval = args.log_interval
    save_images = args.save_images
    train(root, num_epochs, batch_size, device, code_size,
                    log_interval)
