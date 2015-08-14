"""Convenience functions for plotting."""

import numpy as np

__all__ = ['split_color_axis']


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
