import math
import os

import numpy as np
import splitfolders
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_loader(args, augment_strategy='custom', plot_augments=False):
    assert augment_strategy in ['custom', 'rand_augment', 'auto_augment'], \
        'augment_strategy should be in {custom, rand_augment, auto_augment}'

    # mean, std = get_mean_std()
    mean, std = (0.3762, 0.4258, 0.4292), (0.2522, 0.2464, 0.2827)

    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize(mean=mean, std=std) if not plot_augments else nn.Identity()

    # Create augmentation given strategy
    if augment_strategy == 'custom':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(192, 192), scale=(0.1, 1.), interpolation=interpolation),
            transforms.RandomRotation(degrees=(0, 90)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=0.9),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_strategy == 'rand_augment':
        rand_augment = rand_augment_transform(config_str='rand-m10', hparams={'interpolation': Image.BICUBIC})
        train_transform = transforms.Compose([
            transforms.Resize(size=(192, 192), interpolation=interpolation, max_size=None, antialias=None),
            rand_augment,
            transforms.ToTensor(),
            normalize
        ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=interpolation, max_size=None, antialias=None),
        transforms.ToTensor(),
        normalize
    ])

    train_set = datasets.ImageFolder('split_dataset/train', transform=train_transform)
    test_set = datasets.ImageFolder('split_dataset/test', transform=test_transform)

    # Visualize augmentation on samples of the dataset
    if plot_augments:
        for i in range(len(train_set)):
            plot_tensor_img(train_set[i])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, shuffle=False)

    return train_loader, test_loader


def set_model(args, train_all_params=True):
    model = timm.create_model(model_name=args.model, pretrained=True)
    # model = timm.create_model(model_name='efficientnetv2_s', pretrained=True)

    # Set grad true to params for training
    if train_all_params:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    # Update classifier given number of classes
    model.classifier = nn.Linear(in_features=1280, out_features=10, bias=True)

    # Print model
    model_summary(model)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    print('timm model cfg', model.default_cfg)
    return model, criterion


def set_optimizer(args, model):
    if args.optimizer == 'SGD':
        args.lr = 0.05
        args.momentum = 0.9
        args.weight_decay = 0.0001
        args.lr_decay_rate = 0.5
        args.lr_decay_epochs = [5, 10, 25, 50, 100, 150]

        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        args.lr = 0.0001
        args.weight_decay = 0.00001
        args.lr_decay_rate = 0.1
        args.lr_decay_epochs = [50, 75, 100]

        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

    elif args.optimizer == 'RMSprop':
        args.lr = 0.0001
        args.weight_decay = 0.00001
        args.lr_decay_rate = 0.5
        args.lr_decay_epochs = [50, 75, 100]

        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    print(optimizer)

    return optimizer


def model_summary(model):
    print(model)
    print(f'Model size: {sum([param.nelement() for param in model.parameters()]) / 1000000} (M)')


def split_image_dataset(input_folder='dataset', output_folder='split_dataset', ratio=(.8, .0, .2)):
    try:
        assert os.path.isdir(input_folder)

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=ratio)
    except OSError as e:
        print(e)


def plot_tensor_img(dataset_entry):
    img, label = dataset_entry

    np_img = img.detach().cpu().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))

    fig, axs = plt.subplots(figsize=(12, 8))
    axs.imshow(np_img)
    plt.show()


def get_mean_std():
    dataset = datasets.ImageFolder('dataset', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_model(args, model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
