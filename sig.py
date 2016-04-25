import numpy as np
from sklearn.preprocessing import StandardScaler
from mne.filter import low_pass_filter
import statsmodels.api as sm


def car(signal, grouping=None, exclude_elecs=None, agg_func=np.mean,
        normalize=False, return_averages=False):
    """Compute the common average reference of a signal.

    Parameters
    ----------
        signal : array_like, timepoints x features
            the signal to be re-referenced,
        grouping : vector, n_features long
            a group identity vector that is n_features long
        exclude_elecs : array-like
            Any electrode indices to exclude from the common signal
        agg_func : function that takes a vector and returns a scalar
            The function to aggregate across channels
        normalize : bool
            Normalize all channels before computing the average?
        return_averages : bool
            If True, this will return the aggregated signal from each
            group on top of the rereferenced signal.

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
    if normalize is True:
        print('Normalizing signals before CAR')
        scl = StandardScaler()
        signal_car = scl.fit_transform(signal_car)

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
        commonsignal = agg_func(signal_car[:, mask_good], axis=1)
        signal_car[:, mask] -= commonsignal[:, None]
        common_signals[:, i] = commonsignal

    if normalize is True:
        # Return normalized signals to original mean/std
        signal_car = scl.inverse_transform(signal_car)
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
    for i, b in enumerate(snd):
        if on == False:
            if b == True:
                on = True
                start = i
        if on is True:
            if b == True:
                continue
            elif any(snd[i: i+win]):
                # If there sound in the near future, keep going
                continue
            else:
                on = False
                stop = i
                sounds.append([start, stop])
    sounds = np.array(sounds).astype(float) / sfreq
    return sounds


def rms(arr):
    """Calculate the Root Mean Squared power."""
    arr = arr ** 2
    rms = np.sqrt(np.mean(arr))
    return rms


def decimate_nophase(arr, sfreq, q, **kwargs):
    """Decimate a signal without shifting the phase.

    Parameters
    ----------
    arr : numpy array, shape (..., n_times)
        The data we'll decimate.
    sfreq : float | int
        The sampling frequency of the signal.
    q : integer, factor of sfreq
        How much to decimate the signal
    kwargs : passed to mne low_pass_filter function.

    Returns
    -------
    arr : np.array, shape (..., n_times / q).
        The decimated data
    """
    # Calculate downsampling parameters
    arr = np.atleast_2d(arr)
    if np.mod(sfreq, q) != 0:
        raise ValueError('q must be a factor of sfreq')
    new_sfreq = sfreq / q

    # Perform the low-pass filter then decimate manually
    arr = low_pass_filter(arr, sfreq, new_sfreq / 2., **kwargs)
    arr = arr[..., ::q]
    return arr


def compress_signal(sig, kind='log', fac=None):
    '''
    Parameters
    ----------
    sig : array_like
        the signal to compress
    kind : string, one of ['log', 'exp', 'sig'], or None
        Whether we use log-compression, exponential compression,
        or sigmoidal compression. If None, then do nothing.
    fac : float, int
        The factor for the sigmoid if we're using that kind of compression
        Or the root for the exponent if exponential compression.

    Returns
    -------
    out : array, shape(sig)
        The compressed signal
    '''
    # Compression
    if kind == 'sig':
        out = sigp.sigmoid(sig, fac=fac)
    elif kind == 'log':
        out = np.log(sig)
    elif kind == 'exp':
        comp = lambda x, n: x**(1. / n)
        out = comp(sig, fac)
    elif kind is None:
        out = sig
    else:
        raise Exception('You need to specify the correct kind of compression')
    return out


def coh_to_bits(coh):
    """Convert coherence values to a measure of bits."""
    return -np.log2(1-coh)


def remove_1_over_f(psd, freqs=None):
    """Fit a line with robust regression and return the residuals.

    Parameters
    ----------
    psd : array, shape (n_frequencies,)
        The array of PSDs to fit a regression to. Should be log power so that
        a linear model is a proper fit.
    freqs : array, shape (n_frequencies,)
        The frequencies corresponding to each item in psd. If None, a linear
        array of len(psd) will be created

    Returns
    -------
    psd_resid : array, shape (n_frequencies,)
        The residuals after subtracting a linear model fit on the log PSD.
    """
    n_x = psd.shape[0]
    X = np.arange(n_x) if freqs is None else freqs
    mod = sm.RLM(psd, sm.add_constant(X)).fit()
    iint, icoef = mod.params
    psd_resid = psd - (icoef * X + iint)
    return psd_resid
