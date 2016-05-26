# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import mne
from itertools import combinations
from nose.tools import assert_true, assert_raises, assert_equal
from numpy.testing import assert_allclose
from ecogtools.connectivity import (phase_amplitude_coupling,
                                    phase_locked_amplitude,
                                    phase_binned_amplitude)
from sklearn.preprocessing import scale
from pacpy import pac

pac_func = 'plv'


def _create_rand_data():
    """Quickly create some random data."""
    # Set params
    sfreq = 1000.
    n_sig = 3
    n_ep = 8
    n_t = 20.  # Time in one trial
    ev = np.linspace(0, sfreq * n_t, n_ep).astype(int)
    ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).T
    t = np.linspace(0, n_t, sfreq * n_t)

    ixs_conn = np.array(list(combinations(range(n_sig), 2)))
    info = mne.create_info(n_sig, sfreq, 'eeg')

    # Test random signals
    ev = np.vstack([np.linspace(0, len(t), n_ep),
                    np.zeros(n_ep),
                    np.ones(n_ep)]).astype(int).T
    rng = np.random.RandomState(42)
    rand_data = rng.randn(n_ep, n_sig, len(t))
    rand_raw = mne.io.RawArray(np.hstack(rand_data), info)
    rand_epochs = mne.Epochs(rand_raw, ev, {'ev': 1}, -1, 8, preload=True)
    return rand_raw, rand_epochs, ev, ixs_conn


def test_phase_amplitude_coupling():
    """ Test phase amplitude coupling. """
    from scipy.signal import hilbert
    flo = [4, 8]
    fhi = [80, 150]
    eptmin, eptmax = 1, 5

    rand_raw, rand_epochs, ev, ixs_conn = _create_rand_data()
    assert_raises(ValueError,
                  phase_amplitude_coupling, rand_epochs, flo, fhi, ixs_conn)

    rand_raw_test = rand_raw.crop(0, 15, copy=True)  # To speed things up
    conn = phase_amplitude_coupling(
        rand_raw_test, flo, fhi, ixs_conn, pac_func=pac_func)
    assert_true(conn.mean() < .2)

    # Test events handling
    conn = phase_amplitude_coupling(rand_raw, flo, fhi, ixs_conn, ev=ev[:, 0],
                                    tmin=eptmin, tmax=eptmax, pac_func=pac_func)
    assert_true(conn.mean() < .2)
    # events ndim > 1
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, tmin=eptmin, tmax=eptmax, ev=ev, pac_func=pac_func)
    # No tmin/tmax
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, ev=ev, pac_func=pac_func)

    # Test low frequency carrier / modulated oscillation
    n_t = 10.
    sfreq = 1000.
    t = np.linspace(0, n_t, sfreq * n_t)
    lo = np.sin(t * 2 * np.pi * 6)
    hi = np.sin(t * 2 * np.pi * 100)
    ev = np.linspace(0, sfreq * n_t - sfreq * eptmax, 4).astype(int)

    # Clip one signal so it only exists on certain phases of the cycle
    hi[np.angle(hilbert(lo)) > -np.pi * .5] = 0

    # Create Raw array for testing
    data = np.vstack([lo, hi])
    info = mne.create_info(['lo', 'hi'], sfreq, 'eeg')
    data_raw = mne.io.RawArray(data, info)
    conn = phase_amplitude_coupling(
        data_raw, flo, fhi, [0, 1], pac_func=pac_func)
    assert_true(conn > .98)

    # Check that outputs are comparable to pacpy
    conn = phase_amplitude_coupling(
        data_raw, flo, fhi, [0, 1], pac_func=pac_func)
    conn_pacpy = getattr(pac, pac_func)(data_raw._data[0], data_raw._data[1],
                                        flo, fhi, fs=data_raw.info['sfreq'])
    assert_allclose(conn, conn_pacpy, .01)

    # Now noisify the signal and see if this lowers the PAC
    data_raw_noise = data_raw.copy()
    noise_level = 10 * data_raw_noise._data.max()
    data_raw_noise._data += noise_level * np.random.randn(*data_raw_noise._data.shape)
    conn = phase_amplitude_coupling(
        data_raw_noise, flo, fhi, [0, 1], pac_func=pac_func)
    assert_true(conn < .2)

    # Check that scaling doesn't break things
    conn = phase_amplitude_coupling(
        data_raw, flo, fhi, [0, 1], pac_func=pac_func, scale_amp_func=scale)

    # Tests for Raw + events functionality
    conn = phase_amplitude_coupling(data_raw, flo, fhi, [0, 1],
                                    ev=ev, tmin=eptmin, tmax=eptmax,
                                    pac_func=pac_func)
    assert_true(conn > .98)

    conn = phase_amplitude_coupling(data_raw, flo, fhi, [0, 1],
                                    ev=ev, tmin=eptmin, tmax=eptmax,
                                    pac_func=pac_func, concat_epochs=False)
    assert_true(conn.shape[0] > 1)
    assert_true(conn.mean(0) > .98)

    # Check that arrays don't work and correct ixs/freqs must be given
    assert_raises(
        ValueError, phase_amplitude_coupling, data, flo, fhi,
        [0, 1], pac_func=pac_func)
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, fhi,
        [0], pac_func=pac_func)
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo,
        [1], [0, 1], pac_func=pac_func)
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, fhi,
        [0, 1], pac_func='blah')


def test_phase_amplitude_viz_funcs():
    """Test helper functions for visualization"""
    freqs_ph = np.linspace(8, 12, 2)
    freqs_amp = np.linspace(40, 60, 5)
    rand_raw, rand_epochs, ev, ixs_conn = _create_rand_data()

    # Phase locked viz
    ix_ph, ix_amp = [ixs_conn[0][i] for i in [0, 1]]
    amp, phase, times = phase_locked_amplitude(
        rand_epochs, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    amp, phase, times = phase_locked_amplitude(
        rand_raw, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    use_times = rand_raw.times < 3
    amp, phase, times = phase_locked_amplitude(
        rand_raw, freqs_ph, freqs_amp, ix_ph, ix_amp, mask_times=use_times,
        tmin=-.5, tmax=.5)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    # Phase binning
    amp_binned, bins = phase_binned_amplitude(rand_epochs, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(rand_raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(rand_raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20,
                                              mask_times=use_times)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

if __name__ == '__main__':
    test_phase_amplitude_coupling()


# for method in ['plv', 'glm', 'ozkurt']:
#     print(getattr(pac, method)(data[0], data[1], flo, fhi, fs=info['sfreq']))
#     print(phase_amplitude_coupling(data_raw, flo, fhi, [0, 1], pac_func=method)) 
