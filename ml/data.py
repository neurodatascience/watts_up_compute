import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from kaggle_dataset import BrainSegmentationDataset as Dataset
from kaggle_transform import custom_transforms


def get_cifar10_dataset(data_path, batch_size, transform=True):
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

    return train_loader, test_loader

def get_kaggle_dataset(data_path,image_size=256,aug_scale=0.05,aug_angle=15,batch_size=4,workers=4):
    dataset_train = Dataset(
        images_dir = data_path,
        subset = "train",
        transform = custom_transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )

    dataset_test = Dataset(
        images_dir = data_path,
        subset = "validation",
        image_size = image_size,
        random_sampling = False,
    )

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers= workers,
        worker_init_fn=worker_init,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size= batch_size,
        drop_last=False,
        num_workers= workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_test