from .data import un_norm
from .data import classes
import matplotlib.pyplot as plt
from matplotlib import cm


def view_classification(img, ps):
    """View a CIFAR image and it's predicted classes."""
    orig_img = un_norm(img).permute(1, 2, 0)

    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 17), ncols=2)
    ax1.imshow(orig_img, interpolation='bicubic')
    ax1.axis('off')

    ax2.bar(range(10), ps)
    ax2.set_aspect(7.5)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(classes, rotation=45)
    ax2.set_title('Class Probability')
    ax2.set_ylim(0, 1.1)
    plt.tight_layout()
