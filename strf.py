from __future__ import division
import numpy as np
from sklearn.cross_validation import KFold, LabelShuffleSplit, LeavePLabelOut
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from mne.utils import _time_mask
from mne.connectivity import spectral_connectivity
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from .utils import embed
from tqdm import tqdm
from copy import deepcopy

__all__ = ['svd_clean',
           'EncodingModel',
           'delay_timeseries',
           'snr_epochs']


class EncodingModel(object):
    def __init__(self, delays=None, est=None, scorer=None, preproc_y=True):
        """Fit a STRF model.

        Fit a receptive field using time lags and a custom estimator or
        pipeline. This implementation uses Ridge regression and scikit-learn.
        It creates time lags for the input matrix, then does cross validation
        to fit a STRF model.

        Parameters
        ----------
        delays : array, shape (n_delays,)
            The delays to include when creating time lags. The input array X
            will end up having shape (n_feats * n_delays, n_times)
        est : list instance of sklearn estimator | pipeline with estimator
            The estimator to use for fitting. This may be a pipeline, in which
            case the final estimator must create a `coef_` attribute after
            fitting. If an estimator is passed, it also must produce a `coef_`
            attribute after fitting. If estimator is type `GridSearchCV`, then
            a grid search will be performed on each CV iteration (using the cv
            object stored in GridSearchCV). Extra attributes will be generated.
            (see `fit` documentation)
        scorer : function | None
            The scorer to use when evaluating on the held-out test set.
            It must accept two 1-d arrays as inputs (the true values first,
            and predicted values second), and output a scalar value.
            If None, it will be mean squared error.
        preproc_y : bool
            Whether to apply the preprocessing steps of the estimator used in
            fitting on the predictor variables prior to model fitting.
        """
        self.delays = np.array([0]) if delays is None else delays
        self.n_delays = len(self.delays)
        self.est = Ridge() if est is None else est
        self.scorer = mean_squared_error if scorer is None else scorer
        self.preproc_y = preproc_y

    def fit(self, X, y, sfreq, times=None, tmin=None, tmax=None, cv=None,
            preproc_y=False, cv_params=None, feat_names=None, verbose=False):
        """Fit the model.

        Fits a receptive field model. Model results are stored as attributes.

        Parameters
        ----------
        X : array, shape (n_epochs, n_feats, n_times)
            The input data for the regression
        y : array, shape (n_epochs, n_times,)
            The output data for the regression
        sfreq : float
            The sampling frequency for the time dimension
        times : array, shape (n_times,)
            The times corresponding to the final axis of x/y. Is used to
            specify subsets of time per trial (using tmin/tmax)
        tmin : float | array, shape (n_epochs,)
            The beginning time for each epoch. Optionally a different time
            for each epoch may be provided.
        tmax : float | array, shape (n_epochs,)
            The end time for each epoch. Optionally a different time for each
            epoch may be provided.
        cv : int | instance of (KFold, LabelShuffleSplit)
            The cross validation object to use for the outer loop
        feat_names : list of strings/ints/floats, shape (n_feats,) : None
            A list of values corresponding to input features. Useful for
            keeping track of the coefficients in the model after time lagging.
        verbose : bool
            If True, will display a progress bar during fits for CVs remaining.

        Attributes
        ----------
        coef_ : array, shape (n_features * n_lags)
            The average coefficients across CV splits
        coefs_all_ : array, shape(n_cv, n_features * n_lags)
            The raw coefficients for each iteration of cross-validation.
        coef_names : array, shape (n_features * n_lags, 2)
            A list of coefficient names, useful for keeping track of time lags
        scores_ : array, shape (n_cv,)
            Prediction scores for each cross-validation split on the held-out
            test set. Scores are outputs of the `scorer` attribute function.
        best_estimators_ : list of estimators, shape (n_cv,)
            If initial estimator is type `GridSearchCV`, this is the list of
            chosen estimators on each cv split.
        best_params_ : list of dicts, shape (n_cv,)
            If initial estimator is type `GridSearchCV`, this is the list of
            chosen parameters on each cv split.
        """
        if feat_names is not None:
            if len(feat_names) != X.shape[1]:
                raise ValueError(
                    'feat_names and X.shape[0] must be the same size')
        if times is None:
            times = np.arange(X.shape[-1]) / float(sfreq)
        self.tmin = times[0] if tmin is None else tmin
        self.tmax = times[-1] if tmax is None else tmax
        self.times = times
        self.sfreq = sfreq

        # Delay X
        X, y, labels, names = _build_design_matrix(X, y, sfreq, self.times,
                                                   self.delays, self.tmin,
                                                   self.tmax, feat_names)
        self.feat_names = np.array(names)
        cv = _check_cv(X, labels, cv, cv_params)

        # Define names for input variabels to keep track of time delays
        X_names = [(feat, delay)
                   for delay in self.delays for feat in self.feat_names]
        self.coef_names = np.array(X_names)

        # Build model instance
        if not isinstance(self.est, Pipeline):
            self.est = Pipeline([('est', self.est)])

        # Create model metadata that we'll add to the obj later
        model_data = dict(coefs_all_=[], scores_=[])
        if isinstance(self.est.steps[-1][-1], GridSearchCV):
            model_data.update(dict(best_estimators_=[], best_params_=[]))

        # Fit the model and collect model results
        if verbose is True:
            cv = tqdm(cv)
        for i, (tr, tt) in enumerate(cv):
            X_tr = X[:, tr].T
            X_tt = X[:, tt].T
            y_tr = y[tr]
            y_tt = y[tt]

            if self.preproc_y:
                y_tr, y_tt = [self.est._pre_transform(i)[0] for i in [y_tr, y_tt]]
            self.est.fit(X_tr, y_tr)

            mod = deepcopy(self.est.steps[-1][-1])
            if isinstance(mod, GridSearchCV):
                # If it's a GridSearch, then add a "best_params" object
                # Assume hyperparameter search
                if mod.refit:
                    model_data['best_estimators_'].append(mod.best_estimator_)
                    model_data['coefs_all_'].append(mod.best_estimator_.coef_)
                model_data['best_params_'].append(mod.best_params_)
            else:
                model_data['coefs_all_'].append(mod.coef_)

            # Fit model + make predictions
            scr = self.scorer(y_tt, self.est.predict(X_tt))
            model_data['scores_'].append(scr)

        for key, val in model_data.iteritems():
            setattr(self, key, np.array(val))
        self.coefs_ = np.mean(self.coefs_all_, axis=0)
        self.cv = cv

    def predict(self, X):
        """Generate predictions using a fit receptive field model.

        This uses the `coef_` attribute for predictions.
        """
        X_lag = delay_timeseries(X, self.sfreq, self.delays)

        Xt = self.est._pre_transform(X_lag.T)[0]
        return np.dot(Xt, self.coefs_)

    def coefs_as_series(self, agg=None):
        """Return the raw coefficients as a pandas series.

        Parameters
        ----------
        agg : None | function
            If agg is None, all coefs across CVs will be returned. If it
            is a function, it will be applied across CVs and the output
            will be shape (n_coefficients,).

        Outputs
        -------
        sr : pandas Series, shape (n_coefficients,) | (n_cv * n_coefficients)
            The coefficients as a pandas series object.
        """
        ix = pd.MultiIndex.from_tuples(self.coef_names, names=['feat', 'lag'])
        if agg is None:
            sr = []
            for icv, icoef in enumerate(self.coefs_all_):
                isr = pd.DataFrame(icoef[:, np.newaxis], index=ix)
                isr['cv'] = icv
                isr = isr.set_index('cv', append=True).squeeze()
                sr.append(isr)
            sr = pd.concat(sr, axis=0)
        else:
            coefs = agg(self.coefs_all_, axis=0)
            sr = pd.Series(coefs, index=ix)
        return sr

    def plot_coefficients(self, agg=None, ax=None, cmap=None,
                          interpolation='nearest', aspect='auto', **kwargs):
        """Plot the coefficients as a 2D heatmap.

        The plot will be shape (n_features, n_lags)
        """
        from matplotlib import pyplot as plt
        cmap = plt.cm.RdBu_r if cmap is None else cmap
        agg = np.mean if agg is None else agg
        if ax is None:
            f, ax = plt.subplots()
        df = self.coefs_as_series(agg=agg).unstack('lag')
        im = ax.imshow(df.values, cmap=cmap, interpolation=interpolation,
                       aspect=aspect, **kwargs)

        for lab in ax.get_xticklabels():
            lab.set_text(df.columns[int(lab.get_position()[0])])

        for lab in ax.get_yticklabels():
            lab.set_text(df.index[int(lab.get_position()[1])])

        ax.set_xlabel('Time delays (s)')
        ax.set_ylabel('Features')
        return ax


def delay_timeseries(ts, sfreq, delays):
    """Include time-lags for a timeseries.

    Parameters
    ----------
    ts: array, shape(n_feats, n_times)
        The timeseries to delay
    sfreq: int
        The sampling frequency of the series
    delays: list of floats
        The time (in seconds) of each delay. Positive means
        timepoints in the past, negative means timepoints in
        the future.

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


def _scorer_corr(x, y):
    return np.corrcoef(x, y)[1, 0]


def _check_time(X, time):
    if isinstance(time, (int, float)):
        time = np.repeat(time, X.shape[0])
    elif time.shape[0] != X.shape[0]:
        raise ValueError('time lims and X must have the same shape')
    return time


def _check_inputs(X, y, times, delays, tmin, tmax):
    # Add an epochs dimension
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    if y.ndim == 1:
        y = y[np.newaxis, ...]

    if not X.shape[-1] == y.shape[-1] == times.shape[-1]:
        raise ValueError('X, y, or times have different time dimension')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have different number of epochs')
    tmin = _check_time(X, tmin)
    tmax = _check_time(X, tmax)
    if any([np.min(tmin) + np.max(delays) < np.min(times),
            np.max(tmax) + np.min(delays) > np.max(times)]):
        raise ValueError('Data will be cut off w delays, use longer epochs')
    return X, y, tmin, tmax


def _build_design_matrix(X, y, sfreq, times, delays, tmin, tmax, names):
    X, y, tmin, tmax = _check_inputs(X, y, times, delays, tmin, tmax)
    if names is None:
        names = [str(i) for i in range(X.shape[1])]

    # Iterate through epochs with custom tmin/tmax if necessary
    X_out, y_out, lab_out = [[] for _ in range(3)]
    for i, (epX, epy, itmin, itmax) in enumerate(zip(X, y, tmin, tmax)):
        # Create delays
        epX_del = delay_timeseries(epX, sfreq, delays)

        # pull times of interest
        msk_time = _time_mask(times, itmin, itmax)
        epX_out = epX_del[:, msk_time]
        epy_out = epy[msk_time]

        # Unique labels for this epoch
        ep_lab = np.repeat(i + 1, epy_out.shape[-1])

        X_out.append(epX_out)
        y_out.append(epy_out)
        lab_out.append(ep_lab)
    return np.hstack(X_out), np.hstack(y_out), np.hstack(lab_out), names


def _check_cv(X, labels, cv, cv_params):
    cv = 5 if cv is None else cv
    cv_params = dict() if cv_params is None else cv_params
    if isinstance(cv, float):
        raise ValueError('cv must be an int or instance of sklearn cv')

    if len(np.unique(labels)) == 1:
        # Assume single continuous data, cv must take a single number
        if isinstance(cv, int):
            cv_params = dict(n_folds=cv)
            cv = KFold
        cv = cv(labels.shape[-1], **cv_params)
    else:
        # Assume trials structure, cv must take a set of labels for trials
        if isinstance(cv, int):
            cv_params = dict(n_iter=cv)
            cv = LabelShuffleSplit
        cv = cv(labels, **cv_params)
    return cv


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
    from scipy.linalg import svd
    U, s, Vh = svd(arr, full_matrices=False)
    if kind == 'perc':
        s_scaled = s / np.sum(s)
        s_cumulative = np.cumsum(s_scaled)
        s_cut = np.argwhere(s_cumulative > svd_num)[0]
        if s_cut == 0:
            s_cut += 1
        svd_num = range(s_cut)

    U = U[:, svd_num]
    s = np.diag(s.squeeze()[svd_num])
    Vh = Vh[svd_num, :]
    clean_arr = np.dot(U, s).dot(Vh)
    return clean_arr


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
    mps_freqs : array
        The values corresponding to spectral modulations, in cycles / octave
        or cycles / Hz depending on the units of fstep
    mps_times : array
        The values corresponding to temporal modulations, in Hz
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


def snr_epochs(epochs, n_perm=10, fmin=1, fmax=300, tmin=None, tmax=None,
               kind='coh', normalize_coherence=False):
    '''
    Computes the coherence between the mean of subsets of epochs. This can
    be used to assess signal stability in response to a stimulus (repeated or
    otherwise).

    Parameters
    ----------
    epochs : instance of Epochs
        The data on which to calculate coherence. Coherence will be calculated
        between the mean of random splits of trials
    n_perm : int
        The number of permuatations to run
    fmin : float
        The minimum coherence frequency
    fmax : float
        The maximum coherence frequency
    tmin : float
        Start time for coherence estimation
    tmax : float
        Stop time for coherence estimation
    kind : 'coh' | 'corr'
        Specifies the similarity statistic.
        If corr, calculate correlation between the mean of subsets of epochs.
        If coh, then calculate the coherence.
    normalize_coherence : bool
        If True, subtract the grand mean coherence across permutations and
        channels from the output matrix. This is a way to "baseline" your
        coherence values to show deviations from the global means

    Outputs
    -------
    permutations : np.array, shape (n_perms, n_signals, n_freqs)
        A collection of coherence values for each permutation.

    coh_freqs : np.array, shape (n_freqs,)
        The frequency values in the coherence analysis
    '''
    sfreq = epochs.info['sfreq']
    epochs = epochs.crop(tmin, tmax, copy=True)
    nep, n_chan, ntime = epochs._data.shape

    # Run permutations
    permutations = []
    for iperm in tqdm(xrange(n_perm)):
        # Split our epochs into two random groups, take mean of each
        t1, t2 = np.split(np.random.permutation(np.arange(nep)), [nep/2.])
        mn1, mn2 = [epochs[this_ixs]._data.mean(0)
                    for this_ixs in [t1, t2]]

        # Now compute similarity between the two
        this_similarity = []
        for ch, this_mean1, this_mean2 in zip(epochs.ch_names, mn1, mn2):
            this_means = np.vstack([this_mean1, this_mean2])
            if kind == 'coh':
                # import ecogtools as et; et.embed()
                this_means = this_means[np.newaxis, :, :]
                ixs = ([0], [1])
                sim, coh_freqs, _, _, _ = spectral_connectivity(
                    this_means, sfreq=sfreq, method='coh', fmin=fmin,
                    fmax=fmax, tmin=tmin, tmax=tmax, indices=ixs, verbose=0)
                sim = sim.squeeze()
            elif kind == 'corr':
                sim, _ = pearsonr(*this_means)
            else:
                raise ValueError('Unknown similarity type: {0}'.format(kind))
            this_similarity.append(sim)
        permutations.append(this_similarity)
    permutations = np.array(permutations)

    if normalize_coherence is True:
        # Normalize coherence values to their grand average
        permutations -= permutations.mean((0, 1))

    if kind == 'coh':
        return permutations, coh_freqs
    elif kind == 'corr':
        return permutations


def calculate_upper_bound(similarity, n_ep):
    """Calculate an upper bound on model performance.

    Parameters
    ----------
    similarity : array, any shape
        The estimated similarity across epochs as estimated by snr_epochs. This
        is either the correlation or coherence between means of subsets of
        trials for one electrode.
    n_ep : int
        The number of epochs used in the coherence estimation

    Returns
    -------
    upper_bound : array, shape == similarity.shape
        The upper bound on model performance as estimated by coherence across
        trials. This is either the expected coherence or expected correlation,
        depending on which type of input is given in similarity.
    """
    right_hand = .5 * (-n_ep + n_ep * np.sqrt(1. / similarity))
    upper_bound = 1. / (right_hand + 1)
    return upper_bound
