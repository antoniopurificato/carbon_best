import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Subset
import numpy as np

def get_dataset(
    dataset_name: str,
    num_classes: int = 10,
    sample_percentage: float = 1.0,
    specific_classes: list = None,
):
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),  # Repeat channel to convert 1-channel to 3-channel
            ]
        )
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        input_channels = 3
        num_classes = 10

    elif dataset_name == "food101":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = torchvision.datasets.Food101(
            root="./data", download=True, transform=transform
        )
        input_channels = 3
        num_classes = 101

    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        input_channels = 3
        num_classes = 10

    elif dataset_name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        input_channels = 3
        num_classes = 100

    else:
        raise ValueError(
            "Dataset not supported. Choose 'mnist', 'cifar10', or 'cifar100'."
        )

    # Select specific classes if specified
    if specific_classes is not None:
        indices = [
            i for i, label in enumerate(dataset.targets) if label in specific_classes
        ]
        num_classes = len(specific_classes)
        dataset = Subset(dataset, indices)

    return dataset, input_channels, num_classes

def remove_samples(dataset, percentage):
    if percentage is not None:
        class_counts = defaultdict(list)
        try:
            targets = dataset.targets
        except AttributeError:
            targets = dataset._labels
        for idx, label in enumerate(targets):
            if isinstance(label, int):
                class_counts[label].append(idx)
            else:
                class_counts[label.item()].append(idx)
        selected_indices = []
        for class_label, indices in class_counts.items():
            num_samples = int(len(indices) * percentage)
            selected_indices.extend(
                np.random.choice(indices, num_samples, replace=False)
            )
        dataset = Subset(dataset, selected_indices)
    return dataset