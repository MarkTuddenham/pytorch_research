"""Generic util functions."""
import torch as th
from torch.nn import functional as F
from tqdm import tqdm
from collections import OrderedDict

from functools import partial

# from datetime import datetime


def printmd(s):
    from IPython.display import Markdown, display
    display(Markdown(s))


def get_device(no_cuda=False):
    """Get the torch device."""
    cuda = not no_cuda and th.cuda.is_available()
    d = th.device("cuda" if cuda else "cpu")
    print(f'Running on: {th.cuda.get_device_name(d)}')

    return d


def flatten(lst):
    """Flatten a 2D array."""
    return [item for sublist in lst for item in sublist]


def for_each_param(model, f):
    return th.cat([
        f(p)
        for p in model.parameters(recurse=True)
        if p.requires_grad])


clone_gradients = partial(for_each_param,
        f=lambda p: p.grad.clone().detach().flatten())
get_gradients = partial(for_each_param,
        f=lambda p: p.grad.flatten())
clone_weights = partial(for_each_param,
        f=lambda p: p.clone().detach().flatten())
get_weights = partial(for_each_param,
        f=lambda p: p.flatten())


def metrics(loss, logits, labels):
    preds = th.argmax(logits, dim=1)
    acc = (labels == preds).float().mean()
    num_labels = max(max(labels), max(preds))
    cm = th.zeros((num_labels+1, num_labels+1), device=loss.device)
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        # print(f'label: {label.long()}, pred: {pred.long()}')
        cm[label.long(), pred.long()] += 1

    tp = cm.diagonal()[1:].sum()
    fp = cm[:, 1:].sum() - tp
    fn = cm[1:, :].sum() - tp
    return {'loss': loss, 'acc': acc, 'tp': tp, 'fp': fp, 'fn': fn}


def f1_score(tp, fp, fn):
    prec_rec_f1 = {}
    prec_rec_f1['precision'] = tp / (tp + fp)
    prec_rec_f1['recall'] = tp / (tp + fn)
    prec_rec_f1['f1_score'] = 2 * \
            (prec_rec_f1['precision'] * prec_rec_f1['recall']) / \
            (prec_rec_f1['precision'] + prec_rec_f1['recall'])
    return prec_rec_f1


def get_full_loss(model, testloader, device='cuda'):
    predictions = th.zeros(0).to(device)
    labels = th.zeros(0).type(th.LongTensor).to(device)
    model = model.to(device)

    for X, y in iter(testloader):
        y_pred = model(X.to(device))
        predictions = th.cat((predictions, y_pred))
        labels = th.cat((labels, y.to(device)))

    return F.cross_entropy(predictions, labels)


def get_hessian(model, loss, device='cuda'):
    model = model.to(device)

    w = [p for p in model.parameters(recurse=True)
         if p.requires_grad]

    # define here in case too big to fit in memory, then computation not wasted
    size = sum([th.tensor(t.shape).prod() for t in w])
    print(f'Hessian Size: {size}x{size}')
    hessian = th.zeros(size, size).to('cpu')

    w_grad = th.autograd.grad(loss, w, create_graph=True)

    i = 0
    for w_grad_batch in tqdm(w_grad):
        for g in tqdm(w_grad_batch.flatten(), leave=False):
            g2 = th.autograd.grad(g, w, retain_graph=True)
            hessian[i, :] = th.cat([t.flatten() for t in g2]).to('cpu')
            i += 1

    return hessian.detach()


def get_model_param_details(
        model,
        input_size,
        batch_size=-1,
        device=th.device('cuda:0'),
        dtypes=None):
    """Adapted from https://github.com/sksq96/pytorch-summary/blob/011b2bd0ec7153d5842c1b37d1944fc6a7bf5feb/torchsummary/torchsummary.py"""
    if dtypes is None:
        dtypes = [th.FloatTensor]*len(input_size)

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += th.prod(th.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += th.prod(th.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, th.nn.Sequential)
            and not isinstance(module, th.nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [th.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary

# def print(*args, **kwargs):
#     """Custom print function that adds a time signature."""
#     __builtins__.print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]',
#                        end=' ')
#     return __builtins__.print(*args, **kwargs)
