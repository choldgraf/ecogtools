"""Testing the STRF code."""
from ecogtools.strf import EncodingModel, delay_timeseries, snr_epochs
import numpy as np
from sklearn.cross_validation import KFold, LabelShuffleSplit, LeavePLabelOut
from sklearn.linear_model import Ridge
from mne.utils import _time_mask
import mne
import ecogtools as et
import os.path as op

from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises, assert_equal

base_dir = op.join(op.dirname(mne.__file__), 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
rng = np.random.RandomState(42)


def test_encodingmodel():
    """Test the encodingmodel fitting."""
    # Define data
    n_time = 3
    t_start = -.5
    sfreq = 1000
    n_channels = 5
    n_epochs = 10
    times = np.arange(n_time * sfreq) / float(sfreq) + t_start
    delays = np.arange(0, .4, .02)

    # Fitting parameters
    est = Ridge()
    n_iter = 4
    tmin_fit = 0
    tmax_fit = 1.5
    kws_fit = dict(times=times, tmin=tmin_fit, tmax=tmax_fit)
    msk_time = _time_mask(times, tmin_fit, tmax_fit)

    weights = 10 * rng.randn(n_channels * len(delays))
    X = rng.randn(n_epochs, n_channels, n_time * sfreq)
    y = np.stack([np.dot(weights, delay_timeseries(xep, sfreq, delays))
                  for xep in X])

    # --- Epochs data ---
    enc = EncodingModel(delays, est)
    enc.fit(X, y, sfreq, **kws_fit)

    # Make sure CV object and model is correct
    assert_true(isinstance(enc.cv, LabelShuffleSplit))
    assert_equal(enc.cv.labels.shape[-1],
                 np.hstack(y[..., msk_time]).shape[-1])
    assert_true(isinstance(enc.est.steps[-1][-1], type(est)))

    # Epochs w/ custom CV
    cv = LabelShuffleSplit
    cv_params = dict(n_iter=n_iter, test_size=.1)
    enc = EncodingModel(delays, est)
    enc.fit(X, y, sfreq, cv=cv, cv_params=cv_params, **kws_fit)
    assert_true(isinstance(enc.cv, LabelShuffleSplit))
    assert_equal(enc.cv.n_iter, n_iter)
    assert_equal(enc.cv.test_size, .1)

    # Make sure coefficients are correct
    assert_array_almost_equal(weights, enc.coefs_, decimal=2)
    assert_equal(enc.coefs_all_.shape[0], len(enc.cv))

    # Test incorrect inputs
    assert_raises(ValueError, enc.fit, X, y[:2], sfreq)
    assert_raises(ValueError, enc.fit, X, y[..., :5], sfreq)
    assert_raises(ValueError, enc.fit, X, y, sfreq, times=np.array([2, 3]))
    assert_raises(ValueError, enc.fit, X, y, sfreq,
                  tmin=0, tmax=np.array([1, 2]))

    # Test custom tstart / tstop for epochs
    tstarts = .2 * np.random.rand(n_epochs) - tmin_fit
    tstops = .2 * np.random.rand(n_epochs) + tmax_fit
    time_masks = np.array([_time_mask(times, itmin, itmax)
                          for itmin, itmax in zip(tstarts, tstops)])

    enc.fit(X, y, sfreq, times=times, tmin=tstarts, tmax=tstops)
    assert_equal(len(enc.cv.labels), time_masks.sum())

    # Giving time values outside of proper bounds
    assert_raises(ValueError, enc.fit, X, y, sfreq,
                  times=times, tmin=-2, tmax=0)
    assert_raises(ValueError, enc.fit, X, y, sfreq,
                  times=times, tmin=0, tmax=4)

    tstops[5] = 5
    assert_raises(ValueError, enc.fit, X, y, sfreq,
                  times=times, tmin=tstarts, tmax=tstops)

    # --- Single trial data ---
    enc.fit(X[0], y[0], sfreq, **kws_fit)

    # Make sure the CV was chosen correctly + has right time points
    assert_true(isinstance(enc.cv, KFold))
    assert_equal(enc.cv.n, times[msk_time].shape[-1])

    # Loosening the weight requirement because less data
    assert_array_almost_equal(weights, enc.coefs_, decimal=1)


def test_snr():
    """Test trial to trial coherence"""
    raw = mne.io.Raw(raw_fname)
    sfreq = int(raw.info['sfreq'])
    data, times = raw[0, :5 * sfreq]

    # Create fake epochs from copies of the raw + noise
    n_epochs = 40
    noise_amp = .01 * data.max()
    data = np.tile(data, [n_epochs, 1, 1])
    data += noise_amp * rng.randn(*data.shape)
    info = mne.create_info(['ch1'], raw.info['sfreq'], 'eeg')
    ev = np.vstack([np.arange(n_epochs),
                    np.zeros(n_epochs),
                    np.ones(n_epochs)]).T.astype(int)
    epochs = mne.epochs.EpochsArray(data, info, ev)

    # Test CC
    cc = snr_epochs(epochs, kind='corr')
    assert_true((cc > .99).all())

    # Test coherence
    coh, freqs = snr_epochs(epochs, fmin=2, kind='coh')
    assert_true((coh.mean(-1) > .99).all())

    # Test random signal
    data_rand = 10*rng.randn(*data.shape)
    epochs_rand = mne.epochs.EpochsArray(data_rand, info, ev)
    cc = snr_epochs(epochs_rand, kind='corr')
    assert_true(cc.mean() < .02)

    # Test incorrect inputs
    assert_raises(ValueError, snr_epochs, epochs, kind='foo')

if __name__ == '__main__':
    test_snr()
