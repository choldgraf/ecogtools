"""Convenience functions for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from seaborn.palettes import diverging_palette
from seaborn import husl
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

__all__ = ['split_plot_by_color', 'add_rotated_axis', 'AnimatedScatter',
           'plot_activity_on_brain', 'diverging_palette_from_hex',
           'plot_split_circles']


def diverging_palette_from_hex(h1, h2, as_cmap=True):
    """Create a diverging palette from two hex codes."""
    act_h, act_s, act_l = zip(*[husl.hex_to_husl(i) for i in [h1, h2]])
    palette = diverging_palette(*act_h, s=np.mean(act_s),
                                l=np.mean(act_l), as_cmap=as_cmap)
    return palette


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
        pal = diverging_palette_from_hex(*cols, as_cmap=True)
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
                           vmax=None, ax=None, cmap=None, name=None):
    """Plot activity as a scatterplot on a brain.

    Parameters
    ----------
    x : array, shape (n_channels,)
        The x positions of electrodes
    y : array, shape (n_channels,)
        The y positions of electrodes
    act : array, shape (n_channels,)
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

    Returns
    -------
    ax : axis
        The axis object for the plot
    """
    # Handle defaults
    if ax is None:
        _, ax = plt.subplots()
    if cmap is None:
        cmap = plt.cm.coolwarm
    vmin = act.min() if vmin is None else vmin
    vmax = act.max() if vmax is None else vmax

    # Define colors + sizes
    act_norm = (act - vmin) / float(vmax - vmin)
    colors = cmap(act_norm)
    sizes = np.clip(np.abs(act) / float(vmax), 0, 1)  # Normalize to b/w 0 and 1
    sizes = MinMaxScaler((smin, smax)).fit_transform(sizes[:, np.newaxis])

    # Plotting
    ax.imshow(im)
    ax.scatter(x, y, s=sizes, c=colors, cmap=cmap)
    ax.set_title(name, fontsize=20)
    return ax


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


# CFC Viz
def plot_phase_locked_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, mask_times=None,
                                normalize=True,
                                tmin=-.5, tmax=.5, return_data=False,
                                amp_kwargs=None, ph_kwargs=None):
    """Make a phase-locked amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    normalize : bool
        Whether amplitudes are normalized before averaging together. Helps
        if some frequencies have a larger mean amplitude than others.
    tmin : float
        The time to include before each phase peak
    tmax : float
        The time to include after each phase peak
    return_data : bool
        If True, the amplitude/frequency data will be returned
    amp_kwargs : dict
        kwargs to be passed to pcolormesh for amplitudes
    ph_kwargs : dict
        kwargs to be passed to the line plot for phase

    Returns
    -------
    axs : array of matplotlib axes
        The axes used for plotting.
    """
    from .connectivity import phase_locked_amplitude
    from sklearn.preprocessing import scale
    amp_kwargs = dict() if amp_kwargs is None else amp_kwargs
    ph_kwargs = dict() if ph_kwargs is None else ph_kwargs

    data_amp, data_phase, times = phase_locked_amplitude(
        epochs, freqs_phase, freqs_amp,
        ix_ph, ix_amp, tmin=tmin, tmax=tmax, mask_times=mask_times)

    if normalize is True:
        # Scale within freqs across time
        data_amp = scale(data_amp, axis=-1)

    # Plotting
    f, axs = plt.subplots(2, 1)
    ax = axs[0]
    ax.pcolormesh(times, freqs_amp, data_amp, **amp_kwargs)

    ax = axs[1]
    ax.plot(times, data_phase, **ph_kwargs)

    plt.setp(axs, xlim=(times[0], times[-1]))
    if return_data is True:
        return ax, data_amp, data_phase
    else:
        return ax


def plot_phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, normalize=True,
                                n_bins=20, return_data=False,
                                mask_times=None, ax=None,
                                **kwargs):
    """Make a circular phase-binned amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    normalize : bool
        Whether amplitudes are normalized before averaging together. Helps
        if some frequencies have a larger mean amplitude than others.
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size (2 * np.pi) / n_bins.
    return_data : bool
        If True, the amplitude/frequency data will be returned
    ax : matplotlib axis | None
        If not None, plotting functions will be called on this object
    kwargs : dict
        kwargs to be passed to plt.bar

    Returns
    -------
    ax : matplotlib axis
        The axis used for plotting.
    """
    from .connectivity import phase_binned_amplitude
    from sklearn.preprocessing import MinMaxScaler
    amps, bins = phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                        ix_ph, ix_amp, n_bins=n_bins,
                                        mask_times=mask_times)
    if normalize is True:
        amps = MinMaxScaler().fit_transform(amps)
    if ax is None:
        plt.figure()
        ax = plt.subplot(111, polar=True)
    bins_plt = bins[:-1]  # Because there is 1 more bins than amps
    width = 2 * np.pi / len(bins_plt)
    ax.bar(bins_plt + np.pi, amps, color='r', width=width)
    if return_data is True:
        return ax, amps, bins
    else:
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
