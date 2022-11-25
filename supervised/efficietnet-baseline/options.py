import argparse
import math
import os


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

    # optimization
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'],
                        help='optimizers')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosineannealinglr', help='steplr, cos, etc.')
    parser.add_argument('--lr_min', type=float, default=0.00001, help='min. lr')
    parser.add_argument('--lr_warmup_method', type=str, default='none', help='warm up method')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--lr_warmup_decay', type=float, default=0.01, help='warmup decay')

    parser.add_argument('--weight_decay', type=float, default=2e-05, help='regularization')
    parser.add_argument('--norm_weight_decay', type=float, default=0.0, help='regularization')

    parser.add_argument('--model_ema', type=bool, default=True, help='exponential moving average')
    parser.add_argument('--model_ema_steps', type=int, default=32, help='ema steps')
    parser.add_argument('--model_ema_decay', type=float, default=0.99998, help='ema decay')

    parser.add_argument('--label_smoothing', type=float, default=0.1, help='CEL label smothing')

    # augmentation
    parser.add_argument("--auto-augment", default='ta_wide', type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")

    parser.add_argument(
        "--interpolation", default="bicubic", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )


    # model dataset
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_b0')
    parser.add_argument('--dataset', type=str, default='oged', help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'], help='choose method')

    # anomaly detection setting
    parser.add_argument('--normal_class', type=int, default=0, help='normal class on the dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    opt.model_path = './save/{}_models'.format(opt.dataset)

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}__trial_{}'. \
        format(opt.dataset, opt.model, opt.lr,
               opt.weight_decay, opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.lr * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.lr - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.lr

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
