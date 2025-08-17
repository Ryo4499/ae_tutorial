from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST

from .utils.entity import SupportedDataset


def gen_datasets(dataset: SupportedDataset, output_dir: str):
    match dataset:
        case SupportedDataset.MNIST:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            train = MNIST(output_dir, True, transform, download=True)
            test = MNIST(output_dir, False, transform, download=True)
        case SupportedDataset.CIFAR10:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train = CIFAR10(output_dir, True, transform, download=True)
            test = CIFAR10(output_dir, False, transform, download=True)
        case _:
            raise ValueError(f"{dataset} is not supported.")
    train, val = random_split(dataset=train, lengths=[0.9, 0.1])
    return train, val, test


def gen_loaders(dataset: SupportedDataset, output_dir: str, batch_size: int):
    train_dataset, val_dataset, test_dataset = gen_datasets(dataset, output_dir)
    train = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val = DataLoader(
        val_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train, val, test
