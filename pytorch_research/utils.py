"""Generic util functions."""
import torch as th


def setup(args):
    """Apply common torch params and get the device to run on."""
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.enabled = True
    th.backends.cudnn.deterministic = False
    th.manual_seed(args['seed'])

    args['cuda'] = not args['no_cuda'] and th.cuda.is_available()
    device = th.device("cuda" if args['cuda'] else "cpu")
    print(f'Running on: {th.cuda.get_device_name(device)}')

    return device


def flatten(l):
    """Flatten a 2D array."""
    return [item for sublist in l for item in sublist]


def l2_norm(v, sum_dim=0):
    return th.sqrt(th.sum(th.pow(v, 2), dim=sum_dim))


def get_gradients(model):
    return th.cat([
        p.grad.clone().detach().flatten()
        for p in model.parameters(recurse=True)
        if p.requires_grad])


def get_weights(model):
    return th.cat([
        p.clone().detach().flatten()
        for p in model.parameters(recurse=True)
        if p.requires_grad])
