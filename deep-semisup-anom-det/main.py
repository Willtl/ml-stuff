import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import data
import utils
from network import MLP

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def pretrain(net, loader, epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    mse = nn.MSELoss()

    for i in range(epochs):
        n_batches, total_loss = 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = mse(inputs, outputs)
            total_loss += loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Pretrain epoch: {i + 1}, mean loss: {total_loss / n_batches}')

    c = utils.init_center_c(loader, net, device)

    return c


def train(net, loader, c, epochs=1, eta=1.0, eps=1e-9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    milestones = [int(epochs * 0.75), int(epochs * 0.9)]
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.75), int(epochs * 0.9)], gamma=0.1)

    net.train()
    for epoch in range(epochs):
        if epoch in milestones:
            print()

        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net.encode(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            losses = torch.where(targets == 0, dist, eta * ((dist + eps) ** targets.float()))
            loss = torch.mean(losses)

            opt.zero_grad()
            loss.backward()
            opt.step()

            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        print(
            f'| Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')

    return c


def test(net, loader, c):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    scores = torch.zeros(size=(len(loader.dataset),), dtype=torch.float32, device=device)
    labels = torch.zeros(size=(len(loader.dataset),), dtype=torch.long, device=device)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net.encode(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            c_targets = torch.where(targets == 1, 0, 1)

            scores[i * 64: i * 64 + 64] = dist
            labels[i * 64: i * 64 + 64] = c_targets

    scores = scores.to('cpu').numpy()
    labels = labels.to('cpu').numpy()

    auroc = utils.roc(labels, scores, plot=False)
    print(f'AUROC: {auroc}')
    return auroc


def main():
    epochs_pre, epochs_tra = 15, 400

    for i in range(0, 8):
        dataset = f'dataset{i + 1}'

        # Simple MLP with a symmetric decoder for pretraining
        net = MLP(input_size=2, num_features=512, rep_dim=16)

        pretrain_loader = data.get_pretrain(dataset=dataset)
        c = pretrain(net, pretrain_loader, epochs=epochs_pre)

        # Generate data, create datasets and dataloaders
        test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)

        # Pretrain, compute c, and train network
        train(net, train_loader, c, epochs=epochs_tra)

        # Test network and plot ROC
        auroc = test(net, test_loader, c)

        # utils.scores_landscape(net, train_samples, c)
        utils.scores_contour(net, c, test_samples, test_targets, auroc, dataset)


if __name__ == '__main__':
    main()
