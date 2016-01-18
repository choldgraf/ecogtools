from __future__ import division
import numpy as np
from scipy.linalg import svd
from .sig import delay_timeseries
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, LabelShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from .utils import embed

__all__ = ['svd_clean',
           'mps',
           'fit_strf']


def _cluster_mask(msk):
    clust_id = 1
    mask_on = False
    ixs_clusters = np.zeros_like(msk, dtype=int)
    for i, ix in enumerate(msk):
        if ix:
            if not mask_on:
                mask_on = True
            ixs_clusters[i] = clust_id
        elif not ix:
            if mask_on:
                mask_on = False
                clust_id += 1
    return ixs_clusters


def _times_to_mask(times, sfreq, newlen=None):
    time_ixs = (times * sfreq).astype(int)
    masklen = np.max(time_ixs) if newlen is None else newlen
    mask_time = np.zeros(masklen, dtype=bool)
    for istt, istp in time_ixs:
        mask_time[istt:istp] = True
    return mask_time


def _scorer_corr(x, y):
    return np.corrcoef(x, y)[1, 0]


def fit_strf(X, y, sfreq, delays, time_mask=None, alphas=1, cv_outer=None,
             cv_inner=None, scorer_outer=_scorer_corr, X_names=None,
             scale_data=True):
    """Fit a STRF model.

    This implementation uses Ridge regression and scikit-learn. It creates time
    lags for the input matrix, then does cross validation to fit a STRF model.

    Parameters
    ----------
    X : array, shape (n_feats, n_times)
        The input data for the regression
    y : array, shape (n_times,)
        The output data for the regression
    sfreq : float
        The sampling frequency for the time dimension
    delays : array, shape (n_delays,)
        The delays to include when creating time lags. The input array X will
        end up having shape (n_feats * n_delays, n_times)
    time_mask : array, shape (n_times,)
        A mask for only using a subset of timepoints in X/y. Delays must be
        made before a subset of time is taken, thus this allows you to fit a
        model on a subset of time while keeping delays consistent.
    alphas : float | array, shape (n_alphas)
        The alpha values to choose in the Ridge regression. If len(alphas) > 1,
        it corresponds to using an inner CV loop to choose alpha.
    cv_outer : instance of (KFold, LabelShuffleSplit)
        The cross validation object to use for the outer loop
    cv_inner : instance of same type as cv_outer
        The cross validation object to use for the inner loop,
        if len(alphas) > 1
    scorer_outer : function
        The scorer to use when evaluating on the held-out test set.
        It must accept two 1-d arrays as inputs, and output a scalar value.
    X_names : list of strings/ints/floats, shape (n_feats,) : None
        A list of values corresponding to input features. Useful for keeping
        track of the coefficients in the model after time lagging.
    scale_data : bool
        Whether or not to scale the data to 0 mean and unit var before fit.

    Outputs
    -------
    ests : list of sklearn estimators, length (len(cv_outer),)
        The estimator fit on each cv_outer loop. If len(alphas) > 1, then this
        will be the chosen model using GridSearch on each loop.
    scores: np.array, shape (len(cv_outer),)
        The scores on the held out test set on each loop of cv_outer
    ch_names : list of strings, shape (n_feats * n_delays)
        A list of names for each coefficient in the model. It is of structure
        'name_timedelay'.
    """
    # Checks
    alphas = np.atleast_1d(alphas)
    X_names = [str(i) for i in range(len(X))]if X_names is None else X_names
    if len(X_names) != X.shape[0]:
        raise ValueError('X_names and X.shape[0] must be the same size')
    time_mask = np.ones_like(y, dtype=bool) if time_mask is None else time_mask
    if time_mask.shape != y.shape:
        raise ValueError('time_mask and y must be same shape')
    if not X.shape[-1] == y.shape[-1] == time_mask.shape[-1]:
        raise ValueError('X and y and time mask must be same shape')

    # Delay X
    X_delayed = delay_timeseries(X, sfreq, delays)

    # Pull timepoints of interest with time_mask
    if time_mask.dtype is not bool:
        # Assume time mask has cluster indices
        print('Shuffling along cluster indices...')
        clust_ixs = time_mask.copy()
        time_mask = time_mask > 0
        clust_ixs = clust_ixs[time_mask]
        cv_outer.__init__(clust_ixs, n_iter=cv_outer.n_iter,
                          test_size=cv_outer.test_size)
    X_delayed = X_delayed[..., time_mask]
    y = y[..., time_mask]

    if scale_data:
        # Scale features
        print('Scaling features...')
        X_delayed = scale(X_delayed, axis=-1)
        y = scale(y, axis=-1)

    X_names = ['{0}_{1}'.format(feat, delay)
               for delay in delays for feat in X_names]
    if y.shape != cv_outer.labels.shape:
        raise AssertionError("y and cv_outer don't have the same shape") 

    # Fit the models
    ests, scores = [[] for _ in range(2)]

    for i, (tr, tt) in enumerate(cv_outer):
        print('\nCV: {0}'.format(i))
        X_tr = X_delayed[:, tr].T
        X_tt = X_delayed[:, tt].T
        y_tr = y[tr]
        y_tt = y[tt]

        # Set up model and cross-validation
        mod = Ridge(fit_intercept=False, alpha=alphas)
        if len(alphas) > 1:
            if not isinstance(cv_outer, type(cv_inner)):
                raise ValueError('cv objects must be of same type')
            if isinstance(cv_outer, LabelShuffleSplit):
                cv_inner.__init__(cv_outer.labels[tr],
                                  n_iter=cv_inner.n_iter,
                                  test_size=cv_inner.test_size)
            else:
                # Assume kfold
                n_folds = 5 if cv_inner is None else cv_inner.n_folds
                cv_inner = KFold(len(y_tr), n_folds)
            mod = GridSearchCV(mod, param_grid={'alpha': alphas}, cv=cv_inner)

        # Fit model + make predictions
        mod.fit(X_tr, y_tr)
        pred = mod.predict(X_tt)
        scr = scorer_outer(pred, y_tt)

        if len(alphas) > 1:
            print('Alpha scores\n{}'.format(
                '\n'.join([str(s) for s in mod.grid_scores_])))
            mod = mod.best_estimator_
        else:
            print('Score: {0}'.format(scr))

        scores.append(scr)
        ests.append(mod)
    return ests, scores, np.array(X_names)


def svd_clean(arr, svd_num=[0], kind='ix'):
    '''Clean a 2-D array using a Singular Value Decomposition.

    Removes singular values according to either an index number that the user
    provides, or a percentage of variance explained by the singular values. 
    t then transforms data back into original space and returns an array
    with the same index/columns.

    Parameters
    ----------
    arr : pd.DataFrame, n_dim==2
        The array we'll compute SVD on
    svd_num : list of ints, or float
        If kind == 'ix', the indices of the singular values to keep.
        If kind == 'perc', the cutoff percentage of variance explained,
            above which we throw out singular values
    kind : str, ['ix', 'perc']
        See svd_num

    Returns
    -------
    clean_all : np.array
        The cleaned input array
    '''
    U, s, Vh = svd(arr, full_matrices=False)
    if kind == 'perc':
        s_scaled = s / np.sum(s)
        s_cumulative = np.cumsum(s_scaled)
        s_cut = np.argwhere(s_cumulative > svd_num)[0]
        if s_cut == 0: s_cut += 1
        svd_num = range(s_cut)

    U = U[:, svd_num]
    s = np.diag(s.squeeze()[svd_num])
    Vh = Vh[svd_num, :]
    clean_arr = np.dot(U, s).dot(Vh)
    return clean_arr


def create_torc(fmin, fmax, n_freqs, sfreq, duration=1., rip_spec_freq=2,
                rip_temp_freq=2, mod_depth=.9, rip_phase_offset=0,
                time_buffer=.5, combine_waves=True):
    """
    Parameters
    ----------
    fmin : int
        The starting ripple frequency
    fmax : int
        The highest ripple frequency
    n_freqs : int
        The number of log-spaced ripples to simulate between fmin and fmax
    sfreq : int
        The sampling frequency of the ripples (note that fmax must be
        less than sfreq/2)
    duration : float
        The duration of our created ripple stimulus (in seconds)
    rip_spec_freq : float
        How many frequency cycles / octave for the spectral amplitude ripples.
        High values increase ripple frequency as we move upward in
        spectral freq.
    rip_temp_freq : float
        How many temporal cycles / second for the spectral amplitude ripples.
        Positive means ripples have down sweeps, negative means they
        have upsweeps.
        Larger values increase ripple frequency moving forward in time.
    mod_depth : float
        How large will our spectral amplitude modulations be in general.
    rip_phase_offset : float (between 0 and 2pi)
        The starting phase for amplitude modulation.
    time_buffer : float
        How much to buffer the beginning of the ripple.
    combine_waves : bool
        If true, simulated ripple sine waves will be summed together to yield
        a single ripple stimulus

    Outputs
    -------
    output : array, shape (sfreq * duration)
        or shape (n_freqs, sfreq * duration)
        The output ripple stimulus, or the amplitude-modulated sine waves
        (see combine_waves)

    """
    if sfreq / 2 < fmax:
        raise ValueError('fmax is greater than the nyquist frequency')

    # Simulate time and frequencies. Add an extra 100ms for edge effects
    time = np.arange(0, duration+time_buffer, 1/sfreq)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)

    # Create the ripples
    output = np.zeros([freqs.shape[0], time.shape[0]])
    for i, ifreq in enumerate(freqs):
        # Define the amplitude modulation for this sine wave
        ifreq_x = np.log2(ifreq / fmin)
        sin_arg = 2*np.pi * (rip_temp_freq * time + rip_spec_freq * ifreq_x) +\
            rip_phase_offset
        amp = 1 + mod_depth * np.sin(sin_arg)

        # Simulate a sine wave at this frequency, and modulate its amplitude
        wave = np.sin(2*np.pi * time * ifreq)
        wave *= amp
        output[i, :] = wave

    # Combine our sine waves to form a ripple if we want
    if combine_waves is True:
        output = output.sum(0)
    return output[time_buffer*sfreq:]


def mps(strf, fstep, tstep, half=False):
    """Calculate the Modulation Power Spectrum of a STRF.

    Parameters
    ----------
    strf : array, shape (nfreqs, nlags)
        The STRF we'll use for MPS calculation.
    fstep : float
        The step size of the frequency axis for the STRF
    tstep : float
        The step size of the time axis for the STRF.
    half : bool
        Return the top half of the MPS (aka, the Positive
        frequency modulations)

    Returns
    -------
    amps : array
        The MPS of the input strf

    """
    # Convert to frequency space and take amplitude
    nfreqs, nlags = strf.shape
    fstrf = np.fliplr(strf)
    mps = np.fft.fftshift(np.fft.fft2(fstrf))
    amps = np.real(mps * np.conj(mps))

    # Obtain labels for frequency axis
    mps_freqs = np.zeros([nfreqs])
    fcircle = 1.0 / fstep
    for i in range(nfreqs):
        mps_freqs[i] = (i/float(nfreqs))*fcircle
        if mps_freqs[i] > fcircle/2.0:
            mps_freqs[i] -= fcircle

    mps_freqs = np.fft.fftshift(mps_freqs)
    if mps_freqs[0] > 0.0:
        mps_freqs[0] = -mps_freqs[0]

    # Obtain labels for time axis
    fcircle = tstep
    mps_times = np.zeros([nlags])
    for i in range(nlags):
        mps_times[i] = (i/float(nlags))*fcircle
        if mps_times[i] > fcircle/2.0:
            mps_times[i] -= fcircle

    mps_times = np.fft.fftshift(mps_times)
    if mps_times[0] > 0.0:
        mps_times[0] = -mps_times[0]

    if half:
        halfi = np.where(mps_freqs == 0.0)[0][0]
        amps = amps[halfi:, :]
        mps_freqs = mps_freqs[halfi:]

    return mps_freqs, mps_times, amps
