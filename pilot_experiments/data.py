import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append('../')
from ml.kaggle_dataset import BrainSegmentationDataset as Dataset
from ml.kaggle_transform import custom_transforms


def get_cifar10_dataset(data_path, test_cases=None, batch_size=100, transform=True):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                    download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

    # cifar is large enough to ignore early stopping
    # set valid_loader = test_loader to be consistent with other datasets
    valid_loader = test_loader

    return train_loader, valid_loader, test_loader

def get_kaggle_dataset(data_path,test_cases,batch_size=4,image_size=256,subset='all',aug_scale=0.05,aug_angle=15,workers=4):

    if subset in ['train','all']:
        dataset_train = Dataset(
            images_dir = data_path,
            subset = "train",
            validation_cases = test_cases,
            test_cases = test_cases,
            transform = custom_transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
        )

    if subset in ['valid','all']:
        dataset_validation = Dataset(
            images_dir = data_path,
            subset = "validation",
            validation_cases = test_cases,
            test_cases = test_cases,
            image_size = image_size,
            random_sampling = False,
        )
    if subset in ['test','all']:
        dataset_test = Dataset(
            images_dir = data_path,
            subset = "test",
            validation_cases = test_cases,
            test_cases = test_cases,
            image_size = image_size,
            random_sampling = False,
        )

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = None
    loader_validation = None
    loader_test = None

    if subset in ['train','all']:
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers= workers,
            worker_init_fn=worker_init,
        )
    if subset in ['valid','all']:
        loader_validation = torch.utils.data.DataLoader(
            dataset_validation,
            batch_size= batch_size,
            drop_last=False,
            num_workers= workers,
            worker_init_fn=worker_init,
        )
    if subset in ['test','all']:
        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size= batch_size,
            drop_last=False,
            num_workers= workers,
            worker_init_fn=worker_init,
        )

    return loader_train, loader_validation, loader_test