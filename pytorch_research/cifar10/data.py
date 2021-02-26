import torch as th
import torchvision
from torchvision import transforms
from ..cv_utils import DatasetValidationSplitter

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"]


batch_size = 2**7
num_workers=10
# norm_mean = [0.485, 0.456, 0.406]
# norm_std = [0.229, 0.224, 0.225]

norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2023, 0.1994, 0.2010]

normalize = transforms.Normalize(mean=norm_mean,
                                 std=norm_std)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

path = './data'


def get_dataset():
    dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transform_train
    )

    splitter = DatasetValidationSplitter(len(dataset), 0.1)
    train_set = splitter.get_train_dataset(dataset)
    valid_set = splitter.get_val_dataset(dataset)

    test_set = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform_test
    )
    return train_set, valid_set, test_set


def get_train_gen(batch_size=batch_size):
    """Get the generator for the train set."""
    train_set, _, _ = get_dataset()
    return th.utils.data.DataLoader(
        train_set,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


def get_valid_gen(batch_size=batch_size):
    """Get the generator for the validation set."""
    _, valid_set, _ = get_dataset()
    return th.utils.data.DataLoader(
        valid_set,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


def get_test_gen(batch_size=batch_size):
    """Get the generator for the test set."""
    _, _, test_set = get_dataset()
    return th.utils.data.DataLoader(
        test_set,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


un_norm = transforms.Normalize(
    (-th.tensor(norm_mean) / th.tensor(norm_std)).tolist(),
    (1.0 / th.tensor(norm_std)).tolist())
