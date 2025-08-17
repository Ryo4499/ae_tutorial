from enum import StrEnum


class SupportedDataset(StrEnum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class SupportedArchitecture(StrEnum):
    CA = "ca"
    FAA = "faa"
