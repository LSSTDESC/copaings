"""Plot style of the figures: matplotlib settings, blue colour map, colour bar with its label inside."""

import functools

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical colours (Okabe-Ito); LSST, Euclid and HST use the first three
PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#0072B2",
    "#000000",
]

# sequential colour map of the 2D histograms and matrices: very light blue to strong blue
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "copaings_blue", ["#f0f6fc", "#c6dbef", "#2171b5"], N=256
)
mpl.colormaps.register(BLUE_CMAP, force=True)

MY_STYLE = {
    "figure.figsize": [7, 5.25],
    "figure.dpi": 100,
    "savefig.dpi": 500,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "legend.title_fontsize": 12,
    "legend.frameon": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign: draw tick labels with mathtext
    "mathtext.fontset": "cm",
    "axes.prop_cycle": plt.cycler("color", PALETTE),
    "image.cmap": "copaings_blue",
}


def styled(func):
    """Run a plotting function inside `MY_STYLE` (also when it is called directly from a notebook)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(MY_STYLE):
            return func(*args, **kwargs)

    return wrapper


def label_inside_colorbar(cax, label="Count", fontsize=12, y=0.5):
    """Write a label in the middle of a colour bar, white with a black outline (readable on any colour).

    Parameters
    ----------
    cax : matplotlib.axes.Axes
        Axes of the colour bar.
    label : str, default 'Count'
    fontsize : float, default 12
    y : float, default 0.5
        Vertical position of the label centre in the bar (0 = bottom, 1 = top).
    """
    cax.text(
        0.45,
        y,
        label,
        ha="center",
        va="center",
        transform=cax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        fontfamily="sans-serif",
        color="white",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )


def top_colorbar(
    fig,
    mappable,
    axes,
    label="Count",
    height=0.04,
    pad=0.01,
    fontsize=12,
    label_y=0.5,
    side="top",
):
    """Horizontal colour bar above (or below) one or several axes, ticks outside and the label inside the bar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    mappable : matplotlib.cm.ScalarMappable
        Image, hexbin, hist2d quad mesh...
    axes : matplotlib.axes.Axes or list of Axes
        The colour bar spans from the left of the first to the right of the last.
    label : str, default 'Count'
        Text drawn inside the bar (empty: none).
    height, pad : float
        Height of the bar and gap above the axes [figure fraction].
    fontsize : float, default 12
    label_y : float, default 0.5
        Vertical position of the label in the bar (see `label_inside_colorbar`).
    side : {'top', 'bottom'}, default 'top'
        Put the bar above the highest axes or below the lowest ones; the ticks are on that outer side.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    axes = (
        list(axes)
        if isinstance(axes, (list, tuple)) or hasattr(axes, "ravel")
        else [axes]
    )
    for a in axes:
        a.apply_aspect()  # fixed-aspect axes (matshow, set_aspect) shrink only at draw time: get their final box
    boxes = [a.get_position() for a in axes]
    x0 = min([b.x0 for b in boxes])
    x1 = max([b.x1 for b in boxes])
    if side == "top":
        top = max([b.y1 for b in boxes])
        y = top + pad
    else:
        bottom = min([b.y0 for b in boxes])
        y = bottom - pad - height
    cax = fig.add_axes([x0, y, x1 - x0, height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position(side)
    cax.xaxis.set_label_position(side)
    if label:
        label_inside_colorbar(cax, label, fontsize, y=label_y)
    return cbar
