import torch as th
import torchvision
from torchvision import transforms
from ..cv_utils import DatasetValidationSplitter

batch_size = 2**7

norm_mean = [0.1307]
norm_std = [0.3081]

normalize = transforms.Normalize(mean=norm_mean,
                                 std=norm_std)

path = './data'

dataset = torchvision.datasets.MNIST(root=path,
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize
                                     ]))

splitter = DatasetValidationSplitter(len(dataset), 0.1)
train_set = splitter.get_train_dataset(dataset)
valid_set = splitter.get_val_dataset(dataset)

test_set = torchvision.datasets.MNIST(root=path,
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize
                                      ]))


def get_train_gen(batch_size=batch_size):
    """Get the generator for the train set."""
    return th.utils.data.DataLoader(
        train_set,
        # dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=False
    )


def get_valid_gen(batch_size=batch_size):
    """Get the generator for the validation set."""
    return th.utils.data.DataLoader(
        valid_set,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10
    )


def get_test_gen(batch_size=batch_size):
    """Get the generator for the test set."""
    return th.utils.data.DataLoader(
        test_set,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10
    )


un_norm = transforms.Normalize(
    (-th.tensor(norm_mean) / th.tensor(norm_std)).tolist(),
    (1.0 / th.tensor(norm_std)).tolist())
