"""Convenience functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from seaborn.palettes import diverging_palette
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['split_plot_by_color', 'add_rotated_axis', 'AnimatedScatter']


def split_plot_by_color(obj, cutoff=0, cols=[15, 160], clim=None, slim=None):
    '''Color axis data based on whether they're above/below a number.

    Parameters
    ----------
    obj : instance of PathCollection or Rectangle
        The plot data to modify. This is the output of either a scatterplot
        or a histogram
    cutoff : float | int
        The mid point for our color split
    cols : list of ints, length 2
        The colors at the limits of our color range. Should be in HUSL space.
    clim : float
        The width of the window (in data units) around cutoff. Data points
        beyond this window will be saturated.
    slim : list of ints, length 2 | None
        If not None, it must be a list of integers, specifying the min/max
        size of datapoints.

    Returns
    -------
    obj : instance of input
        The modified input object
    '''
    # Define the color palette we'll use
    if isinstance(cols, (list, tuple)):
        pal = diverging_palette(*cols, as_cmap=True)
    elif isinstance(cols, LinearSegmentedColormap):
        pal = cols
    else:
        raise ValueError('Cols must be a list of ints, or colormap')

    if isinstance(obj, list):
        # Histogram
        if isinstance(obj[0], Rectangle):
            for patch in obj:
                if np.round(patch.get_x(), 10) < cutoff:
                    patch.set_color(pal(float(0)))
                else:
                    patch.set_color(pal(float(1)))
        else:
            raise ValueError('List of unknown types provided')
    elif isinstance(obj, PathCollection):
        # Scatterplot
        edgecol = obj.get_edgecolors()
        xy = obj.get_offsets()
        diff = xy[:, 1] - xy[:, 0]
        if clim is not None:
            if not all([isinstance(i, int) for i in cols]):
                raise ValueError('Cols must be int if clim is given')
            clim = np.max(np.abs(diff)) if clim is None else clim
            diff = diff - cutoff
            # Normalize b/w -.5 and .5, then add .5 to be 0-1
            diff = np.clip(diff, -clim, clim) / float(clim*2)
            diff += .5
            obj.set_color(pal(diff))
            if slim is not None:
                smin, smax = slim
                # Renorm size between -.5, .5, then double so its -1 to 1
                sdiff = (np.abs(diff - .5) * 2)
                # Now scale it to the right size / min amount
                sizes = sdiff * (smax - smin) + smin
                obj.set_sizes(sizes)
        else:
            over = diff > cutoff
            lowpal, highpal = [np.tile(pal(i), [len(over), 1]) for i in [0, 1]]
            newcolors = np.where(over[:, None], pal(float(0)), pal(float(1)))
            obj.set_color(newcolors)
        obj.set_edgecolors(edgecol)
    else:
        raise ValueError('Unknown type provided: {0}'.format(type(obj)))
    return obj


def header_ax(s, x=.5, y=.5, fontsize=30, ax=None, **kwargs):
    """Create a matplotlib plot as a header w/ text."""
    if ax is None:
        f, ax = plt.subplots()
    ax.text(x, y, s, fontsize=fontsize, ha='center', va='center', **kwargs)
    ax.axis('off')
    return ax


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