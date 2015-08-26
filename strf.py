from __future__ import division
import numpy as np
from scipy.linalg import svd

__all__ = ['svd_clean']


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
