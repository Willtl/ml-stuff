import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import utils


class CustomDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets
        self.n_samples = samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __len__(self):
        return self.n_samples


def gen_gaussian(n_samples=1000, ratio_normal=0.5, circular=False):
    n_normal = int(n_samples * ratio_normal)
    n_anomal = int(n_samples * (1 - ratio_normal))
    assert n_normal % 2 == 0, 'ratio of normal not divisible by 2'
    assert n_anomal % 2 == 0, 'ratio of normal not divisible by 2'

    norm1 = np.random.randn(n_normal // 2, 2).astype(np.float32)
    norm1[:, 0] -= 3
    norm1[:, 1] += 3
    norm2 = np.random.randn(n_normal // 2, 2).astype(np.float32) * 0.5
    normal = np.concatenate((norm1, norm2))
    # normal = np.random.randn(int(n_samples * ratio_normal), 2).astype(np.float32)
    normal_label = np.ones(normal.shape[0]).astype(np.int)

    anomal = np.random.randn(int(n_samples * (1 - ratio_normal)), 2).astype(np.float32)
    anomal = anomal * 0.3
    anomal[:, 0] += 0.5
    anomal[:, 1] += 2.5
    anomal_label = np.ones(normal.shape[0]).astype(np.int) * -1

    if circular:
        ox, oy = 0, 0
        for i in range(anomal.shape[0]):
            px, py = anomal[i][0], anomal[i][1]
            angle = np.random.random() * (np.pi * 2)
            anomal[i][0] = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            anomal[i][1] = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    samples = np.concatenate((normal, anomal))
    targets = np.concatenate((normal_label, anomal_label))

    return torch.from_numpy(samples), torch.from_numpy(targets)


def get_loaders(dataset='gaussian'):
    if dataset == 'gaussian':
        train_samples, train_targets = gen_gaussian(n_samples=10000, ratio_normal=0.999)
        test_samples, test_targets = gen_gaussian(n_samples=10000, ratio_normal=0.5)

        train_ds = CustomDataset(train_samples, train_targets)
        test_ds = CustomDataset(test_samples, test_targets)
    else:
        samples, targets, test_samples, test_targets = load_data(dataset, ratio_normal=0.2, ratio_anomaly=0.0)

        train_ds = CustomDataset(samples, targets)
        test_ds = CustomDataset(test_samples, test_targets)

        # if True:
        #     utils.plot(samples, targets, 'train_samples', dataset)
        #     utils.plot(test_samples, test_targets, 'test_samples', dataset)

    train_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True, drop_last=True, pin_memory=True,
                              num_workers=1, persistent_workers=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, pin_memory=True)

    return test_samples, test_targets, train_loader, test_loader


def get_pretrain(dataset='gaussian'):
    if dataset == 'gaussian':
        samples, targets = gen_gaussian(n_samples=10000, ratio_normal=1.0)
        train_ds = CustomDataset(samples, targets)
    else:
        train_samples, train_targets, _, _ = load_data(dataset, ratio_normal=0.5, ratio_anomaly=0.0)
        train_ds = CustomDataset(train_samples, train_targets)

    pretrain_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
    return pretrain_loader


def load_data(dataset_name, test_size=0.33, ratio_normal=0.5, ratio_anomaly=1.0, pollution=0.0):
    """ Load data from json

    Attributes:
        dataset_name    name of the dataset in folder datasets
        test_size       ratio of samples for testing
        ratio_normal    ratio of the unlabeled data that is labeled as normal
        ratio_anomaly   ratio of the normal data that is labeled as anomaly
        pollution        not implemented
    """

    with open(f'datasets/{dataset_name}.json') as json_file:
        data = json.load(json_file)
        normal = np.array(data['normal']).astype(np.float32)
        anomaly = np.array(data['anomaly']).astype(np.float32)

        normal, norm_test = train_test_split(normal, test_size=test_size, random_state=42)
        norm_train, unlabeled = train_test_split(normal, test_size=1 - ratio_normal, random_state=42)
        anom_train, anom_test = train_test_split(anomaly, test_size=test_size, random_state=42)

        # Set amount of anomalies given ratio
        amount_anom = int(norm_train.shape[0] * ratio_anomaly)
        assert amount_anom < anomaly.shape[0], 'anomaly ratio error'
        indexes = np.random.choice(anom_train.shape[0], size=amount_anom, replace=False)
        anom_train = anom_train[indexes]

        # Define targets for unlabeled, normal, and anomaly
        unlabeled_train_targets = np.zeros(unlabeled.shape[0]).astype(np.int)
        norm_train_targets = np.ones(norm_train.shape[0]).astype(np.int)
        norm_test_targets = np.ones(norm_test.shape[0]).astype(np.int)
        anom_train_targets = np.ones(anom_train.shape[0]).astype(np.int) * -1
        anom_test_targets = np.ones(anom_test.shape[0]).astype(np.int) * -1

        # Concatenate data
        train_samples = np.concatenate((unlabeled, norm_train, anom_train))
        train_targets = np.concatenate((unlabeled_train_targets, norm_train_targets, anom_train_targets))
        test_samples = np.concatenate((norm_test, anom_test))
        test_targets = np.concatenate((norm_test_targets, anom_test_targets))

        print(f'Training: {unlabeled.shape[0]} unlabeled, {norm_train.shape[0]} normal, '
              f'and {anom_train.shape[0]} anomaly samples')
        print(f'Testing:  {norm_test.shape[0]} normal,'
              f' and {anom_test.shape[0]} anomaly samples')

        return train_samples, train_targets, test_samples, test_targets
