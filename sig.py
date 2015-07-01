import numpy as np
from sklearn.preprocessing import scale


def car(signal, grouping=None, exclude_elecs=None,
        normalize=False, return_averages=False):
    """Compute the common average reference of a signal.

    Parameters
    ----------
        signal : array_like, timepoints x features
            the signal to be re-referenced,
        grouping : vector, n_features long
            a group identity vector that is n_features long

    Returns
    -------
    signal_car : array_like
        the signal after having the common averaged removed
    """
    if grouping is None:
        grouping = np.ones(signal.shape[1])
    else:
        grouping = np.asarray(grouping).copy()

    if exclude_elecs is not None:
        exclude_elecs = np.asarray(exclude_elecs, dtype=bool)

    if grouping.shape[0] != signal.shape[1]:
        raise AssertionError(['Grouping vec must be same'
                             'length as n_features'])

    signal_car = signal.copy()
    if normalize:
        signal_car = scale(signal_car, axis=0)

    groups = np.unique(grouping[grouping > 0])
    print('Number of groups found: {0}'.format(len(groups)))

    common_signals = np.zeros([signal.shape[0], len(groups)])
    for i, group in enumerate(groups):
        # Define a mask for the group, and one for only good chans
        mask = (grouping == group)
        if exclude_elecs is not None:
            mask_good = mask * ~exclude_elecs
        else:
            mask_good = mask

        # Compute the CAR and subtract
        commonsignal = signal_car[:, mask_good].mean(axis=1)
        signal_car[:, mask] -= commonsignal[:, None]
        common_signals[:, i] = commonsignal
    if return_averages:
        return signal_car, common_signals
    else:
        return signal_car


def delay_timeseries(ts, sfreq, delays):
    """Include time-lags for a timeseries.

    Parameters
    ----------
    ts: array, shape(n_feats, n_times)
        The timeseries to delay
    sfreq: int
        The sampling frequency of the series
    delays: list of floats
        The time (in seconds) of each delay
    Returns
    -------
    delayed: array, shape(n_feats*n_delays, n_times)
        The delayed matrix
    """
    delayed = []
    for delay in delays:
        roll_amount = int(delay * sfreq)
        rolled = np.roll(ts, roll_amount, axis=1)
        if delay < 0:
            rolled[:, roll_amount:0] = 0
        elif delay > 0:
            rolled[:, 0:roll_amount] = 0
        delayed.append(rolled)
    delayed = np.vstack(delayed)
    return delayed


def find_sound_times(snd, sfreq, win_size):
    """Find groups of sound times by clustering in time.

    Parameters
    ----------
    snd : array (type bool)
        A boolean array of sound timepoints, often
        created with a cutoff
    sfreq : int
        The sampling frequency of snd
    win_size : float
        The size of the search window in seconds

    Returns
    -------
    sounds : array, shape(n_sounds, 2)
        start and stop times for each sound, in seconds.
    """
    on = False
    win = int(sfreq * win_size)
    sounds = []
    for i, b in enumerate(over):
        if on == False:
            if b == True:
                on = True
                start = i
        if on is True:
            if b == True:
                continue
            elif any(over[i: i+win]):
                continue
            else:
                on = False
                stop = i
                sounds.append([start, stop])
    sounds = np.array(sounds)
    return sounds / float(sfreq)
