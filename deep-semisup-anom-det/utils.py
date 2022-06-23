import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import torch
from cycler import cycler
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mpl.style.use('classic')


def init_center_c(loader, net, device, eps=0.1):
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = net.encode(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f'Computed center: {c}')
    return c


def roc(labels, scores, plot=True):
    fpr = dict()
    tpr = dict()

    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    # new_auc = auroc_score(labels, scores)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if plot:
        # Colors, color cycles, and colormaps
        mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

        # Colormap
        mpl.rcParams['image.cmap'] = 'jet'

        # Grid lines
        mpl.rcParams['grid.color'] = 'k'
        mpl.rcParams['grid.linestyle'] = ':'
        mpl.rcParams['grid.linewidth'] = 0.5

        # Figure size, font size, and screen dpi
        mpl.rcParams['figure.figsize'] = [8.0, 6.0]
        mpl.rcParams['figure.dpi'] = 80
        mpl.rcParams['savefig.dpi'] = 100
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['figure.titlesize'] = 'medium'

        # Marker size for scatter plot
        mpl.rcParams['lines.markersize'] = 3

        # Plot
        mpl.rcParams['lines.linewidth'] = 0.9
        mpl.rcParams['lines.dashed_pattern'] = [6, 6]
        mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
        mpl.rcParams['lines.dotted_pattern'] = [1, 3]
        mpl.rcParams['lines.scale_dashes'] = False

        # Error bar
        mpl.rcParams['errorbar.capsize'] = 3

        # Patch edges and color
        mpl.rcParams['patch.force_edgecolor'] = True
        mpl.rcParams['patch.facecolor'] = 'b'

        plt.figure()
        lw = 1
        plt.plot(fpr, tpr, color='darkorange', label='(AUC = %0.4f, EER = %0.4f)' % (auroc, eer))
        plt.plot([eer], [1 - eer], marker='o', markersize=3, color="navy")
        # plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.plot([0, 0], [0, 1], color='navy', linestyle=':')
        plt.plot([0, 1], [1, 1], color='navy', linestyle=':')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.savefig(f'{args.exp_path}/auc_{epoch}.svg', bbox_inches='tight', format='svg', dpi=800)
        plt.show()
        plt.close()

    return auroc


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        with torch.no_grad():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def scores_landscape(net, train_samples, c, min_input=-10, max_input=10, size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = np.zeros((size, size, 2)).astype(np.float32)
    step = (max_input - min_input) / size
    for i in range(size):
        for j in range(size):
            inputs[i][j][0] = min_input + (step * i) + (step / 2)
            inputs[i][j][1] = min_input + (step * j) + (step / 2)

    inputs = torch.from_numpy(inputs).view(-1, 2)
    scores = torch.zeros(size * size).view(-1)
    net.eval()
    with torch.no_grad():
        for i in range(0, inputs.shape[0] // 64):
            input = inputs[i * 64:i * 64 + 64]
            input = input.to(device)
            outputs = net.encode(input)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            scores[i * 64: i * 64 + 64] = dist

    scores = scores.view(size, size).numpy()

    # train_samples = train_samples.numpy()
    # plt.scatter(train_samples[:, 0], train_samples[:, 1], color='blue')
    plt.imshow(scores.T, cmap='gray', interpolation='bilinear', origin='lower')
    plt.tight_layout()
    plt.colorbar()
    plt.show()


def plot(samples, targets, file, dataset):
    n_unlab, n_norm, n_anom = 0, 0, 0
    for i in range(targets.shape[0]):
        if targets[i] == -1:
            n_anom += 1
        elif targets[i] == 0:
            n_unlab += 1
        else:
            n_norm += 1

    cdict = {-1: 'red', 0: 'grey', 1: 'blue'}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_title(f"# unlabeled: {n_unlab}, # normal: {n_norm}, # anomaly: {n_anom}")

    for g in np.unique(targets):
        if g == -1:
            label = 'Anomaly'
        elif g == 0:
            label = 'Unlabeled'
        else:
            label = "Normal"

        ix = np.where(targets == g)
        ax.scatter(samples[ix, 0], samples[ix, 1], c=cdict[g], label=label, s=30, linewidth=0.4)
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.tight_layout()
    plt.savefig(f'results/{dataset}/{file}', dpi=300)
    # plt.show()


def scores_contour(net, c, test_samples, test_targets, auroc, dataset, add_points=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    zs = np.zeros((100, 100))

    net.eval()
    with torch.no_grad():
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                # zs[i][j] = x[i] ** 2 + y[j] ** 2
                input = torch.tensor([x[i], y[j]], dtype=torch.float32, device=device)
                output = net.encode(input)
                dist = torch.sum((output - c) ** 2)
                zs[i][j] = dist.to('cpu').numpy().item()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    ax.set_title(f"Score's Contours Plot - AUROC: {auroc:.5f}")

    cp = ax.contourf(x, y, np.transpose(zs), 1000)
    cbar = fig.colorbar(cp)
    cbar.set_label('Score', rotation=270, labelpad=15)
    ax.autoscale(False)

    if add_points:
        cdict = {-1: 'red', 0: 'grey', 1: 'blue'}
        for g in np.unique(test_targets):
            label = 'Normal' if g == 1 else 'Anomaly'
            ix = np.where(test_targets == g)
            ax.scatter(test_samples[ix, 0], test_samples[ix, 1], c=cdict[g], label=label, s=30, linewidth=0.4)
        ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig(f'results/{dataset}/scores_contour_unsup', dpi=300)
    # plt.show()
