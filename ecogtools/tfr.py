import mne
from glob import glob
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

__all__ = ['extract_amplitude']


def extract_amplitude(inst, freqs, n_cycles=7, normalize=False, n_hilbert=None,
                      picks=None, n_jobs=1, ):
    """Extract the time-varying amplitude for a frequency band.

    If multiple freqs are given, the amplitude is calculated at each frequency
    and then averaged across frequencies.

    Parameters
    ----------
    inst : instance of Raw
        The data to have amplitude extracted
    freqs : array of ints/floats, shape (n_bands, 2)
        The frequencies to use. If multiple frequency bands given, amplitude
        will be extracted at each and then averaged between frequencies. The
        structure of each band is fmin, fmax.
    n_cycles : int
        The number of cycles to include in the filter length for the hilbert
        transform.
    normalize : bool
        Whether to normalize each frequency amplitude by its mean before
        averaging. This can be helpful if some frequencies have a much higher
        base amplitude than others.
    n_hilbert : int | 'auto' | None
        The length of data to use in the Hilbert transform. The data will be
        cut to last dimension of this size. If 'auto', the length equal to the
        next highest power of two will be used.
    picks : array | None
        The channels to use for extraction

    Returns
    -------
    inst : mne instance, same type as input 'inst'
        The MNE instance with channels replaced with their time-varying
        amplitude for the supplied frequency range.
    """

    # Data checks
    n_hilbert = inst.n_times if n_hilbert is None else n_hilbert
    if n_hilbert == 'auto':
        n_hilbert = int(2 ** np.ceil(np.log2(inst.n_times)))
    n_hilbert = int(n_hilbert)
    freqs = np.atleast_2d(freqs)
    if freqs.shape[-1] != 2:
        raise ValueError('freqs must be shape (n_fbands, 2)')
    picks = range(len(inst.ch_names)) if picks is None else picks

    # Filter for HFB and extract amplitude
    bands = np.zeros([freqs.shape[0], len(inst.ch_names), inst.n_times])
    for i, (fmin, fmax) in enumerate(freqs):
        length_filt = int(np.floor(n_cycles * sfreq / fmin))
        # Extract a range of frequency bands for averaging later
        inst_band = inst.copy()
        inst_band.filter(fmin, fmax, filter_length=length_filt, n_jobs=n_jobs)
        inst_band.apply_hilbert(picks, envelope=True,
                                n_jobs=n_jobs, n_fft=n_hilbert)

        if normalize is True:
            # Scale frequency band so that the ratios of all are the same
            inst_band_mn = inst_band._data.mean(1)[:, np.newaxis]
            inst_band._data /= inst_band_mn
        bands[i] = inst_band._data.copy()

    # Average across fbands
    inst._data[picks, :] = bands.mean(0)
    return inst
