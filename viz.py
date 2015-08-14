"""Convenience functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

__all__ = ['split_color_axis', 'add_rotated_axis']


def split_color_axis(ax, cutoff=0, cols=['g', 'r']):
    '''Color axis data based on whether they're above/below a number.

    Takes an axis object coming from a call to scatter or hist.
    Then, colors all bars lower than "cutoff" to red, and
    others to green.
    '''
    if len(ax.patches) > 0:
        # Histogram
        for patch in ax.patches:
            if np.round(patch.get_x(), 10) < cutoff:
                patch.set_color(cols[1])
            else:
                patch.set_color(cols[0])
    elif len(ax.collections) > 0:
        # Scatterplot
        xy = ax.collections[0].get_offsets()
        diff = xy[:, 1] - xy[:, 0]
        over = diff > cutoff
        ax.collections[0].set_color(np.where(over, *cols))
    else:
        raise ValueError('Axis must have a scatterplot or histogram')


def add_rotated_axis(f, extents=(-1, 1, 0, 1), sc_x=None, sc_y=None,
                     rotation=-45, position=(.5, .5), invisible_border=True):
    """Add a rotated axis to an existing figure.

    Parameters
    ----------
    f : mpl.figure
        The figure you're adding a axis to.
    extents : tuple, shape of ints (4,)
        The x min/max and y min/max of the axis
    sc_x : float
        How much to scale the x-axis for shaping nicely
    sc_y : float
        How much to scale the y_axis for shaping nicely
    rotation : float
        Rotation of the axis
    position : tuple of floats, shape (2,)
        The position of the axis as a fraction of the figure
    invisible_border : bool
        Whether to make elements of the axis invisible

    Returns
    -------
    ax : mpl axis object
        The axis you've added
    ax_aux : mpl auxilliary axis object
        The auxilliary object of the axis you've added. This
        is useful for doing certain kinds of transformations that
        aren't possible with the regular axis.
    """
    af = Affine2D()
    transform = af.scale(sc_x, sc_y).rotate_deg(-45)
    helper = floating_axes.GridHelperCurveLinear(transform, extents)
    ax = floating_axes.FloatingSubplot(f, 111, grid_helper=helper)
    ax_aux = ax.get_aux_axes(transform)
    f.add_subplot(ax)
    ax.set_position(np.hstack([position, .5, .5]))

    if invisible_border is True:
        # Strip axis elements
        ax.patch.set(visible=False)
        for axis in ax.axis.values():
            axis.set_visible(False)

    return ax, ax_aux
