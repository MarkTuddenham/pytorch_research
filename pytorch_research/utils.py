"""Generic util functions."""
import torch as th

from functools import partial

from datetime import datetime


def printmd(s):
    from IPython.display import Markdown, display
    display(Markdown(s))


def get_device(no_cuda=False):
    """Get the torch device."""
    cuda = not no_cuda and th.cuda.is_available()
    d = th.device("cuda" if cuda else "cpu")
    print(f'Running on: {th.cuda.get_device_name(d)}')

    return d


def flatten(l):
    """Flatten a 2D array."""
    return [item for sublist in l for item in sublist]


def for_each_param(model, f):
    return th.cat([
        f(p)
        for p in model.parameters(recurse=True)
        if p.requires_grad])


clone_gradients = partial(for_each_param, f=lambda p: p.grad.clone().detach().flatten())
get_gradients = partial(for_each_param, f=lambda p: p.grad.flatten())
clone_weights = partial(for_each_param, f=lambda p: p.clone().detach().flatten())
get_weights = partial(for_each_param, f=lambda p: p.flatten())


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
    prec_rec_f1['f1_score'] = 2 * (prec_rec_f1['precision'] * prec_rec_f1['recall']
                                   ) / (prec_rec_f1['precision'] + prec_rec_f1['recall'])
    return prec_rec_f1


def get_hessian(model, loss, device='cuda'):
    model = model.to(device)

    w = [p for p in model.parameters(recurse=True)
         if p.requires_grad]

    # define here in case too big to fit in memory, then computation not wasted
    size = sum([th.tensor(t.shape).prod() for t in w])
    print(f'Hessian Size: {size}x{size}')
    hessian = th.zeros(size, size).to('cpu')

    w_grad = th.autograd.grad(loss, w, create_graph=True)

    d2_w = []
    i = 0
    for w_grad_batch in tqdm(w_grad):
        for g in tqdm(w_grad_batch.flatten(), leave=False):
            g2 = th.autograd.grad(g, w, retain_graph=True)
            hessian[i, :] = th.cat([t.flatten() for t in g2]).to('cpu')
            i += 1

    return hessian.detach()


# def print(*args, **kwargs):
#     """Custom print function that adds a time signature."""
#     __builtins__.print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]', end=' ')
#     return __builtins__.print(*args, **kwargs)
