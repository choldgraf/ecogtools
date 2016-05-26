"""Quick utility functions."""

from IPython import embed
import numpy as np
import pandas as pd
import mne
import sys
from os import path, sep, remove
from glob import glob
from datetime import datetime


def vembed():
    import matplotlib.pyplot as plt
    embed()


def ipy_post_mortem():
    """Causes ipython/idb to be called in the event of an error"""
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(
        mode='Verbose', color_scheme='Linux', call_pdb=1)


def add_ix_to_dataframe(df, ix_dict):
    for key, val in ix_dict.items():
        ix = np.repeat(val, df.shape[0])
        ix = pd.Index(ix, name=key)
        df = df.set_index(ix, append=True)
    return df


def apply_across_df_level(df, levels, func=np.mean, verbose=False):
    """Average out a level from a dataframe.

    This is a convenience function to quickly average away a set
    of levels in a dataframe. The dataframe will be grouped by all
    other levels, and then averaged.

    Parameters
    ----------
    df : DataFrame
        The dataframe in question
    levels : list of strings
        The names of the levels to average out

    Returns
    -------
    df : DataFrame
        The input dataframe with levels averaged out
    """
    if not isinstance(levels, (list, tuple)):
        raise ValueError('level names must be in a list')
    count_levels = filter(lambda a: a not in levels, df.index.names)
    if verbose is True:
        print('Retaining levels: {0}'.format(count_levels))
    df = df.groupby(level=count_levels)
    df = df.agg(func)
    return df


def query_from_dicts(dict_list):
    """
    Create a series of string queries from a dictionary.

    This allows you to create a query to be used with pandas .query or .eval
    using dictionaries for quick slicing.

    Parameters
    ----------
    dict_list : list of dict
        The dictionaries to use to build queries. Queries will be constructed
        using key
    """
    if not isinstance(dict_list, (list, np.ndarray)):
        dict_list = [dict_list]
    qu_list = []
    for item in dict_list:
        join_list = []
        for key, val in item.iteritems():
            if isinstance(val, str):
                val = [val]
            join_str = key + ' in {0}'.format(list(val))
            join_list.append(join_str)
        qu_list.append(join_list)
    queries = [' and '.join(items) for items in qu_list]
    if len(queries) == 1:
        queries = queries[0]
    return queries


def bin_and_apply(data, bin_centers, func=np.mean):
    """Aggregate data by bin centers and apply a function to combine

    This will find the nearest bin_center for each value in data,
    group them together by the closest bin, and apply func to the
    values.

    Parameters
    ----------
    data : np.array, shape(n_points,)
        The 1-d data you'll be binning
    bin_centers : np.array, shape(n_bins)
        The centers you compare to each data point
    func : function, returns scalar from 1-d data
        Data within each bin group will be combined
        using this function.

    Returns
    -------
    new_vals : np.array, shape(n_bins)
        The data combined according to bins you supplied."""
    ix_bin = np.digitize(data, bin_centers)
    new_vals = []
    for ibin in np.unique(ix_bin):
        igroup = data[ix_bin == ibin]
        new_vals.append(func(igroup))
    new_vals = np.array(new_vals)
    return(new_vals)


def add_timestamp_to_folder(save_path, append=None, overwrite=True):
    """Add a timestamp to a save folder."""
    dir_name = path.dirname(save_path)
    today = datetime.today()
    time_stamp = '_'.join([str(i) for i in [today.date(),
                                            'h{}'.format(today.hour),
                                            'm{}'.format(today.minute)]])
    if append is not None:
        if overwrite is True:
            to_remove = glob(dir_name + '/*{0}.stamp'.format(append))
            for over_file in to_remove:
                print('Removing file: {0}'.format(over_file))
                remove(over_file)
        time_stamp = time_stamp + '__' + append + '.stamp'
    file_stamp = path.dirname(save_path) + '/{0}'.format(time_stamp)
    open(file_stamp, 'a').close()


def decimate_by_binning(data, data_names, n_decim):
    """Decimates along the first axis."""
    data_names = np.array(data_names).astype(int)

    # Calculate binning paramters
    n_bins = int(data.shape[0] / n_decim)
    bins = np.hstack([i]*n_decim for i in range(n_bins))

    # Do the binning and take mean between them
    data = np.vstack([data[bins == i].mean(0) for i in np.unique(bins)])
    data_names = np.array([int(data_names[bins == i].mean())
                           for i in np.unique(bins)])
    return data, data_names


def zscore_by_non_events(data, times, events, with_mean=True, with_std=True):
    """z-score data excluding timepoints contained in events

    Parameters
    ----------
    data : array, shape (..., n_times)
        The data to z-score, generally of shape (n_signals, n_times)
    times : array, shape (n_times,)
        The times in data
    events : array, shape (n_events, 2)
        The start/stop times (in seconds) of events
    with_mean : bool
        Whether to subtract the mean of non-event times
    with_std : bool
        Whether to divide by standard deviation of non-event times

    Returns
    -------
    data : array, shape == data.shape
        The data after z-scoring
    """
    msk_events = [mne.utils._time_mask(times, istt, istp)
                  for istt, istp in events]
    msk_events = np.vstack(msk_events).any(0)
    msk_non_events = ~msk_events
    if with_mean:
        imn = data[..., msk_non_events].mean(-1)
        data = data - imn[:, np.newaxis]

    if with_std:
        istd = data[..., msk_non_events].std(-1)
        data = data / istd[:, np.newaxis]
    return data
