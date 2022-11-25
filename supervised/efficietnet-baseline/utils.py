import math
import os
import time

import numpy as np
import splitfolders
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from timm.data.auto_augment import rand_augment_transform, auto_augment_transform
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

import presets
import transforms

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


def get_scheduler(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    return main_lr_scheduler


def get_warmup_scheduler(args, main_lr_scheduler, optimizer):
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler


def load_data_with_pytorch_transforms(args, traindir='split_dataset/train', valdir='split_dataset/test', plot=True):
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = args.ra_magnitude
    augmix_severity = args.augmix_severity
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
    )
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # Visualize augmentation on samples of the dataset
    if plot:
        for i in range(len(dataset)):
            plot_tensor_img(dataset[i])

    quit()

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    return train_loader, test_loader


def load_data_with_timm_transforms(args, augment_strategy='custom', plot=False):
    img_sz_train, img_sz_test, crop_pct = 192, 312, 0.875
    interpolation = transforms.InterpolationMode.BICUBIC
    scale_size_test = int(math.floor(img_sz_test / crop_pct))

    # mean, std = get_mean_std()
    mean, std = (0.3762, 0.4258, 0.4292), (0.2522, 0.2464, 0.2827)
    # mean, std = torch.tensor(IMAGENET_DEFAULT_MEAN), torch.tensor(IMAGENET_DEFAULT_STD)
    normalize = transforms.Normalize(mean=mean, std=std) if not plot else nn.Identity()

    # Create augmentation given strategy
    if augment_strategy == 'custom':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_sz_train, scale=(0.1, 1.0), interpolation=interpolation),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=0.9),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_strategy == 'rand':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_sz_train, scale=(0.1, 1.0), interpolation=interpolation),
            rand_augment_transform(config_str='rand-m10-n2-mstd0.5', hparams={'interpolation': Image.BICUBIC}),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_strategy == 'auto':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_sz_train, scale=(0.1, 1.0), interpolation=interpolation),
            auto_augment_transform(config_str='original-mstd0.5', hparams={'interpolation': Image.BICUBIC}),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_strategy == 'imagenet':
        train_transform = transforms_imagenet_train(img_size=img_sz_train)
        test_transform = transforms_imagenet_eval(img_size=img_sz_test)

    if augment_strategy != 'imagenet':
        test_transform = transforms.Compose([
            transforms.Resize(scale_size_test, interpolation=interpolation),
            transforms.CenterCrop(img_sz_test),
            transforms.ToTensor(),
            normalize
        ])

    print('Train transform: ', train_transform)
    print('Test transform: ', test_transform)

    train_set = datasets.ImageFolder('split_dataset/train', transform=train_transform)
    test_set = datasets.ImageFolder('split_dataset/test', transform=test_transform)

    # Visualize augmentation on samples of the dataset
    if plot:
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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    print('timm model cfg', model.default_cfg)
    return model, criterion


def set_optimizer(args, model):
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316,
            alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    # if args.optimizer == 'SGD':
    #     optimizer = optim.SGD(model.parameters(),
    #                           lr=args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    # elif args.optimizer == 'Adam':
    #     args.lr = 0.0001
    #     args.weight_decay = 0.00001
    #     args.lr_decay_rate = 0.1
    #     args.lr_decay_epochs = [50, 75, 100]
    #
    #     optimizer = optim.Adam(model.parameters(),
    #                            lr=args.lr,
    #                            weight_decay=args.weight_decay)
    #
    # elif args.optimizer == 'RMSprop':
    #     args.lr = 0.0001
    #     args.weight_decay = 0.00001
    #     args.lr_decay_rate = 0.5
    #     args.lr_decay_epochs = [50, 75, 100]
    #
    #     optimizer = optim.RMSprop(model.parameters(),
    #                               lr=args.lr,
    #                               momentum=args.momentum,
    #                               weight_decay=args.weight_decay)

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
