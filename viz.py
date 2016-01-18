"""Convenience functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

__all__ = ['split_color_axis', 'add_rotated_axis', 'AnimatedScatter']


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
                     rotation=45, position=(.5, .5, .1, .1),
                     invisible_border=True):
    """Add a rotated axis to an existing figure.

    Parameters
    ----------
    f : mpl.figure
        The figure you're adding a axis to.
    extents : tuple, shape of ints (4,)
        The x min/max and y min/max of the axis, as well as axis data boundaries.
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
    transform = af.scale(sc_x, sc_y).rotate_deg(rotation)
    helper = floating_axes.GridHelperCurveLinear(transform, extents)
    ax = floating_axes.FloatingSubplot(f, 111, grid_helper=helper)
    ax_aux = ax.get_aux_axes(transform)
    f.add_subplot(ax)
    ax.set_position(position)
    ax.invert_xaxis()

    if invisible_border is True:
        # Strip axis elements
        ax.patch.set(visible=False)
        for axis in ax.axis.values():
            axis.set_visible(False)

    return ax, ax_aux


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation.

    Parameters
    ----------
    x : shape(n_features)
    y : hape(n_features)
    s : shape(n_frames, n_features)
    time : n_frames
    interval : int 
        Number of milliseconds between frame draws
    frames : int | array
        How many frames to play, or which frames to play
    im : array
        A background image
    save_path : str
        Saves the movie to this location"""
    def __init__(self, x, y, s, c=None, size_mult=1, time=None, interval=100, frames=None,
                 im=None, save_path=None, ax=None, cmap=None, **kwargs):
        self.x = x
        self.y = y
        self.s = s
        self.c = s if c is None else c
        self.im = im
        self.time = time
        self.interval = interval
        self.frames = s.shape[0] if frames is None else frames
        self.cmap = cmap
        assert(x.shape[0] == y.shape[0] == s.shape[1]), 'Data have unequal number of points'
        
        self.n = self.s.shape[0]
        self.size_mult = size_mult

        # Setup the figure and axes...
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.interval,
                                           frames=self.frames, init_func=self.setup_plot,
                                           blit=True, **kwargs)
        if save_path is not None:
            print('Saving movie to: {0}'.format(save_path))
            self.ani.save(save_path)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        s, c = [self.s[0], self.c[0]]
        self.scat = self.ax.scatter(self.x, self.y, c, animated=True, cmap=self.cmap)
        if self.im is not None:
            self.ax.imshow(self.im)

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, 

    def update(self, i):
        """Update the scatter plot."""
        s = self.s[i]
        c = self.c[i]

        # Set sizes...
        self.scat._sizes = s * self.size_mult

        # Set colors..
        self.scat.set_array(c)

        if self.time is not None:
            self.ax.title.set_text(self.time[i])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.ax.title

    def show(self):
        plt.show()