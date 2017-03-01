"""Convenience functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import seaborn as sns
import colorbabel as cb
sns.set_style('white')


__all__ = ['split_plot_by_color', 'add_rotated_axis',
           'plot_activity_on_brain', 'plot_split_circles', 'set_axis_font']


def split_plot_by_color(obj, cutoff=0, cols=None, clim=None, slim=None,
                        color_cutoffs=None, color_cutoff_value=None):
    '''Color axis data based on whether they're above/below a number.

    Parameters
    ----------
    obj : instance of PathCollection or Rectangle
        The plot data to modify. This is the output of either a scatterplot
        or a histogram
    cutoff : float | int
        The mid point for our color split
    cols : list of hex codes, length 2 | None
        The colors at the limits of our color range. Should be a hex code that
        will be converted to HUSL space.
    clim : float
        The width of the window (in data units) around cutoff. Data points
        beyond this window will be saturated.
    slim : list of ints, length 2 | None
        If not None, it must be a list of integers, specifying the min/max
        size of datapoints.
    color_cutoffs : array, shape (2,)
        A cutoff point for each axis (x, y) below which the data points will
        be turned to cut_color.
    color_cutoff_value : None | array, shape (4,)
        The value to replace colors with if they are below color_cutoffs.
        Must be a length 4 array/list of rgba values.

    Returns
    -------
    obj : instance of input
        The modified input object
    '''
    # Define the color palette we'll use
    cols = ['#67a9cf', '#ef8a62'] if cols is None else cols
    if isinstance(cols, (list, tuple)):
        if not all([isinstance(i, (str)) for i in cols]):
            raise ValueError('Cols must be list of hex codes if not colormap')
        pal = cb.ColorTranslator(cols).to_diverging()
    elif isinstance(cols, LinearSegmentedColormap):
        pal = cols
    else:
        raise ValueError('Cols must be a list of ints/floats, or colormap')

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

        # To grey out points below a certain cutoff
        if color_cutoffs is not None:
            kws_cut_lines = dict(linestyles='--', color='k', alpha=.2)
            if color_cutoff_value is None:
                color_cutoff_value = [.5, .5, .5, .1]
            else:
                if len(color_cutoff_value) != 4:
                    raise ValueError('Color cutoff must be an array of length 4')
            # Get current object properties
            ax = obj.axes
            cut_x, cut_y = color_cutoffs
            offsets = obj.get_offsets()
            colors = obj.get_facecolors()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()

            # Make the offset cut and apply changes
            offsets_cut = np.logical_and(offsets[:, 0] < cut_x,
                                         offsets[:, 1] < cut_y)
            colors[offsets_cut, :] = color_cutoff_value
            obj.set_facecolors(colors)
            obj.set_edgecolors(colors)
            ax.vlines(cut_x, ax.get_ylim()[0], cut_x, **kws_cut_lines)
            ax.hlines(cut_y, ax.get_xlim()[0], cut_y, **kws_cut_lines)
            ax.set(xlim=xlim, ylim=ylim)
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


def plot_activity_on_brain(x, y, act, im, smin=10, smax=100, vmin=None,
                           vmax=None, ax=None, cmap=None, name=None,
                           movie_duration=5.):
    """Plot activity as a scatterplot on a brain.

    Parameters
    ----------
    x : array, shape (n_channels,)
        The x positions of electrodes
    y : array, shape (n_channels,)
        The y positions of electrodes
    act : array, shape (n_channels, [n_times])
        The activity values to plot as size/color on electrodes
    im : ndarray, passed to imshow
        An image of the brain to match with x/y positions
    smin : int
        The minimum size of points
    smax : int
        The maximum size of points
    vmin : float | None
        The minimum color value / size cutoff
    vmax : float | None
        The maximum color value / size cutoff
    ax : axis | None
        An axis object to plot to
    cmap : matplotlib colormap | None
        The colormap to plot
    name : string | None
        A string name for the plot title.
    movie_duration : int | float
        The duration (in seconds) of the movie created if `act`
        has two dimensions.

    Returns
    -------
    ax : axis
        The axis object for the plot
    """
    # Handle defaults
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    if cmap is None:
        cmap = plt.cm.coolwarm
    vmin = act.min() if vmin is None else vmin
    vmax = act.max() if vmax is None else vmax

    # Define colors + sizes
    if act.ndim == 1:
        act = act[:, np.newaxis]
        do_movie = False
    elif act.ndim == 2:
        do_movie = True
    else:
        raise ValueError('`act` must be shape (n_channels, [n_times])')

    # Normalize colors to the vmin/vmax so colors are correct
    act_norm = (act - vmin) / float(vmax - vmin)
    colors = cmap(act_norm)

    # For size, scale normalized activity to between -1 and 1
    sizes = (act_norm * 2) - 1

    # Now take absolute value so 0 is the midpoint
    sizes = np.abs(sizes)

    # Now scale up to the sizes specified
    sizes = sizes * (smax - smin) + smin

    # Plotting
    ax.imshow(im)
    scat = ax.scatter(x, y, s=sizes[:, 0], c=colors[:, 0], cmap=cmap)
    if name is not None:
        ax.set_title(name, fontsize=20)
    ax.set_axis_off()

    # Make it a movie if we wish
    if do_movie is True:
        sfreq_movie = act.shape[-1] / movie_duration

        # Function to update the scatterplot
        def animate_scatterplot(t):
            ix = int(np.round(t * sfreq_movie))
            this_sizes = sizes[:, ix]
            this_colors = colors[:, ix]
            scat.set_sizes(this_sizes)
            scat.set_color(this_colors)
            return mplfig_to_npimage(fig)

        # Now we'll create our videoclip using this function
        clip = VideoClip(animate_scatterplot, duration=movie_duration)
        clip.fps = sfreq_movie
        return clip

    return ax


def plot_split_circles(centers, radius, n_wedges=2, angle=0, ax=None,
                       colors=None, scale_by_ax=True, **kwargs):
    """
    Plot a circle at the specified location that is split into sub-colors.

    Parameters
    ----------
    centers : array, shape (n_circles, 2)
        The x/y coordinate of the circle centers
    radius : float | int
        The radius of the circle. If scale_by_ax is True, must be b/w 0 and 1
    n_wedges : int
        The number of wedges in the circle
    angle : float
        The rotation angle in degrees
    ax : None | mpl axis
        The axis object to plot on. If None, a new axis is created.
    colors : list, length == n_wedges | None
        The colors for each wedge
    scale_by_ax : bool
        Whether to treat `radius` as raw data units, or as a fraction of the
        figure x-axis (in which case radius must be between 0 and 1)
    kwargs : dictionary
        To be passed to the mpl Wedge call.

    Returns
    -------
    wedges : list
        A lit of mpl Wedge objects correspond to each wedge in the circle
    """
    from matplotlib.patches import Wedge
    if ax is None:
        f, ax = plt.subplots()
    if colors is None:
        colors = plt.cm.rainbow(np.arange(n_wedges) / float(n_wedges))
    if scale_by_ax is True:
        radius = radius * (ax.get_xlim()[1] - ax.get_xlim()[0])
    radii = radius
    if isinstance(radii, (int, float)):
        radii = np.repeat(radii, centers.shape[0])

    arc = 360. / n_wedges

    # Do the plotting
    wedges = []
    for ixy, iradius in zip(centers, radii):
        for icolor in colors:
            wedges.append(Wedge(ixy, iradius, angle, angle + arc,
                                fc=icolor, **kwargs))
            angle = angle + arc
    for iwedge in wedges:
        ax.add_artist(iwedge)
    return wedges


def set_axis_font(axs, fontproperties):
    """Set the font of text in an axis using a fontproperties object."""
    axs = np.atleast_1d(axs)
    for ax in axs.ravel():
        props = (ax.get_xticklabels() + ax.get_yticklabels() +
                 [ax.title, ax.xaxis.label, ax.yaxis.label])
        _ = plt.setp(props, fontproperties=fontproperties)


def layout_to_xy(layout, im):
    """Convert an MNE layout to xy points.

    This can be confusing because the layout plots from the top to bottom (to
    match the imshow default), while scatterplots go from bottom to top. As
    such, this function flips the y values, then expands both x and y to fill
    the size of `im`.

    Parameters
    ----------
    layout : instance of mne layout
        The layout corresponding to electrode locations
    im : ndarray
        The image of a brain the electrodes belong to. It should be at least
        2-D and the first two dimensions are assumed to be the Y / X axis.

    Returns
    -------
    xy : array, shape (n_elecs, 2)
        The xy positions of the electrodes in `layout`.
    """
    x, y = layout.pos[:, :2].T
    # Reverse y
    y = 1 - y
    sh_y, sh_x = im.shape[:2]
    xy = np.vstack([x * sh_x, y * sh_y]).T
    return xy


def plot_equation(eq, fontsize=50, outfile=None, padding=0.1, ax=None):
    """Plot an equation as a matplotlib figure.

    Parameters
    ----------
    eq : string
        The equation that you wish to plot. Should be plottable with
        latex. If `$` is included, they will be stripped.
    fontsize : number
        The fontsize passed to plt.text()
    outfile : string
        Name of the file to save the figure to.
    padding : float
        Amount of padding around the equation in inches.
    ax : matplotlib axis | None
        The axis for plotting. If None, a new figure will be created.
        Defaults to None.

    Returns
    -------
    ax : matplotlib axis
        The axis with your equation.
    """
    # clean equation string
    eq = eq.replace('$', '').replace(' ', '')

    # set up figure
    if ax is None:
        f = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
    else:
        f = ax.figure
    r = f.canvas.get_renderer()

    # display equation
    t = ax.text(0.5, 0.5, '${}$'.format(eq), fontsize=fontsize,
                horizontalalignment='center', verticalalignment='center')

    # resize figure to fit equation
    bb = t.get_window_extent(renderer=r)
    w = bb.width / f.dpi
    h = np.ceil(bb.height / f.dpi)
    f.set_size_inches((padding + w, padding + h))

    # set axis limits so equation is centered
    _ = plt.setp(ax, xlim=[0, 1], ylim=[0, 1])

    # Remove formatting for axis
    ax.grid(False)
    ax.set_axis_off()

    # Save and return
    if outfile is not None:
        f.savefig(outfile)
    return ax
