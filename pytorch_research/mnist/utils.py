import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def view_classification(img, ps):
    """Function for viewing an image and it's predicted classes."""
    if ps.dim() == 2:
        fig, axs = plt.subplots(figsize=(9, 14), ncols=2, nrows=(ps.shape[0]+2)//2)
        axs[0][0].imshow(
            img.resize_(1, 28, 28).numpy().squeeze(),
            interpolation='None', cmap=cm.gray)
        axs[0][0].axis('off')
        _, *axs = axs.flatten()
        for i, p in enumerate(ps):
            p = p.cpu().data.numpy().squeeze()
            axs[i].bar(np.arange(10), p)
            axs[i].set_aspect(7.5)
            axs[i].set_xticks(np.arange(10))
            axs[i].set_xticklabels(np.arange(10))
            axs[i].set_title('Class Probability')
            axs[i].set_ylim(0, 1.1)
    else:
        ps = ps.cpu().data.numpy().squeeze()
        fig, (ax1, ax2) = plt.subplots(figsize=(9, 14), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), interpolation='None', cmap=cm.gray)
        ax1.axis('off')
        ax2.bar(np.arange(10), ps)
        ax2.set_aspect(7.5)
        ax2.set_xticks(np.arange(10))
        ax2.set_xticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_ylim(0, 1.1)
    plt.tight_layout()
