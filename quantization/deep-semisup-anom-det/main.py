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

dataset = f'dataset1'
epochs_pre, epochs_tra = 1, 1


def pretrain(net, loader, epochs=1):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    mse = nn.MSELoss()

    for i in range(epochs):
        n_batches, total_loss = 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(utils.device), targets.to(utils.device)
            opt.zero_grad()
            outputs = net.decode(net.encode(inputs))
            loss = mse(inputs, outputs)
            total_loss += loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Pretrain epoch: {i + 1}, mean loss: {total_loss / n_batches}')

    c = utils.init_center_c(loader, net)

    return c


def train(net, loader, c, epochs=1, eta=2.0, eps=1e-9):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    milestones = [int(epochs * 0.75), int(epochs * 0.9)]
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.75), int(epochs * 0.9)], gamma=0.1)

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for inputs, targets in loader:
            inputs, targets = inputs.to(utils.device), targets.to(utils.device)

            outputs = net.encode(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            losses = torch.where(targets == 0, dist, eta * ((dist + eps) ** targets.float()))
            loss = torch.mean(losses)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            scheduler.step()

        if epoch in milestones:
            print(f'Adjusted learning rate: {float(scheduler.get_last_lr()[0]):3}')

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        print(
            f'| Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')

    return c


def test(net, loader, c):
    scores = torch.zeros(size=(len(loader.dataset),), dtype=torch.float32, device=utils.device)
    labels = torch.zeros(size=(len(loader.dataset),), dtype=torch.long, device=utils.device)

    net.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(utils.device), targets.to(utils.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                c_targets = torch.where(targets == 1, 0, 1)

                scores[i * 64: i * 64 + 64] = dist
                labels[i * 64: i * 64 + 64] = c_targets
        end = time.time()
    scores = scores.to('cpu').numpy()
    labels = labels.to('cpu').numpy()

    auroc = utils.roc(labels, scores, plot=False)
    print(f'AUROC: {auroc}, runtime: {end - start}')
    return auroc


# AUROC: 0.968504194322892, time: 3.835312843322754
# AUROC: 0.968866211257929, time: 1.8843259811401367
def post_training_dynamic_quantization():
    # Simple MLP with a symmetric decoder for pretraining
    net = MLP(input_size=2, num_features=5000, rep_dim=256)

    pretrain_loader = data.get_pretrain(dataset=dataset)
    c = pretrain(net, pretrain_loader, epochs=epochs_pre)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)

    # Pretrain, compute c, and train network
    train(net, train_loader, c, epochs=epochs_tra)

    utils.device = 'cpu'
    net = net.to(utils.device)
    c = c.to(utils.device)

    # Test network and plot ROC
    test(net, test_loader, c)

    net_int8 = torch.quantization.quantize_dynamic(
        net,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights

    test(net_int8, test_loader, c)


# AUROC: 0.968504194322892, time: 3.835312843322754
# AUROC: 0.9672603980724038, runtime: 1.0422911643981934
def post_training_static_quantization():
    # Simple MLP with a symmetric decoder for pretraining
    net = MLP(input_size=2, num_features=5000, rep_dim=256).to(utils.device)

    pretrain_loader = data.get_pretrain(dataset=dataset)
    c = pretrain(net, pretrain_loader, epochs=epochs_pre)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)

    # Pretrain, compute c, and train network
    train(net, train_loader, c, epochs=epochs_tra)

    # Move stuff to CPU and set eval mode
    utils.device = 'cpu'
    net = net.to(utils.device)
    net.eval()
    c = c.to(utils.device)

    # Test network and plot ROC
    test(net, test_loader, c)
    print(net.encoder)

    m = net.encoder
    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(m, ['2', '3'], inplace=True)
    m = nn.Sequential(torch.quantization.QuantStub(),
                      *m,
                      torch.quantization.DeQuantStub())
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)

    samples = torch.from_numpy(test_samples)
    with torch.inference_mode():
        for i in range(samples.shape[0]):
            m(samples[i])

    torch.quantization.convert(m, inplace=True)
    test(m, test_loader, c)
    print(m)

    m = torch.jit.trace(m, samples)

    # Test network and plot ROC
    test(m, test_loader, c)


def quantization_aware_training():
    # Simple MLP with a symmetric decoder for pretraining
    net = MLP(input_size=2, num_features=5000, rep_dim=256).to(utils.device)

    pretrain_loader = data.get_pretrain(dataset=dataset)
    c = pretrain(net, pretrain_loader, epochs=epochs_pre)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)

    test(net, test_loader, c)

    m = net.encoder
    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(m, ['2', '3'], inplace=True)
    m = nn.Sequential(torch.quantization.QuantStub(),
                      *m,
                      torch.quantization.DeQuantStub())

    m.train()
    m.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare_qat(m, inplace=True)

    opt = optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-6)
    m.train()
    for epoch in range(epochs_tra):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(utils.device), targets.to(utils.device)

            outputs = m(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            losses = torch.where(targets == 0, dist, 1.0 * ((dist + 1e-9) ** targets.float()))
            loss = torch.mean(losses)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        print(
            f'| Epoch: {epoch + 1:03}/{epochs_tra:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')

    utils.device = 'cpu'
    m.eval()
    m = m.to(utils.device)
    c = c.to(utils.device)

    torch.quantization.convert(m, inplace=True)

    # Move stuff to CPU and set eval mode
    test(m, test_loader, c)


def jit_speedup():
    # Simple MLP with a symmetric decoder for pretraining
    net = MLP(input_size=2, num_features=5000, rep_dim=256)

    pretrain_loader = data.get_pretrain(dataset=dataset)
    c = pretrain(net, pretrain_loader, epochs=epochs_pre)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)

    # Pretrain, compute c, and train network
    train(net, train_loader, c, epochs=epochs_tra)

    utils.device = 'cpu'
    net = net.to(utils.device)
    c = c.to(utils.device)

    test(net, test_loader, c)

    x = torch.from_numpy(test_samples)
    net = torch.jit.trace(net, x)

    # Test network and plot ROC
    test(net, test_loader, c)


def main():
    # post_training_dynamic_quantization()
    post_training_static_quantization()
    # quantization_aware_training()

    # jit_speedup()


if __name__ == '__main__':
    main()
