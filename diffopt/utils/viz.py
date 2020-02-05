import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


LINEWIDTH = 6
STYLE = {
    'g1': dict(label=r"$|g^1_t - g^*|$", color='C0', linewidth=LINEWIDTH),
    'g2': dict(label=r"$|g^2_t - g^*|$", color='C1', linewidth=LINEWIDTH),
    'g3': dict(label=r"$|g^3_t - g^*|$", color='C2', linewidth=LINEWIDTH),
    'z': dict(label=r"$|z_t - z^*|$", color='k', linestyle='--',
              linewidth=LINEWIDTH),
}

# Setup matplotlib fonts
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 24


def color_palette(n_colors=4, cmap='viridis', extrema=False):
    """Create a color palette from a matplotlib color map"""
    if extrema:
        bins = np.linspace(0, 1, n_colors)
    else:
        bins = np.linspace(0, 1, n_colors * 2 - 1 + 2)[1:-1:2]

    cmap = plt.get_cmap(cmap)
    palette = list(map(tuple, cmap(bins)[:, :3]))
    return palette


def make_legend(ax):
    handles = [mpl.lines.Line2D([0], [0], **style) for style in STYLE.values()]
    labels = [style['label'] for style in STYLE.values()]
    ax.set_axis_off()
    ax.legend(handles, labels, loc='center', bbox_to_anchor=(0, .95, 1, .05),
              ncol=4, fontsize=24)
