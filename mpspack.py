import numpy as np
import mne

def mps(strf, fstep, tstep, return_top_half=False, return_amp=True):
    """Calculate the Modulation Power Spectrum of a STRF.

    Parameters
    ----------
    strf : array, shape (nfreqs, nlags)
        The STRF we'll use for MPS calculation.
    fstep : float
        The step size of the frequency axis for the STRF
    tstep : float
        The step size of the time axis for the STRF.
    return_half_mps : bool
        Return the top half of the MPS (aka, the Positive
        frequency modulations)
    return_amp : bool
        Whether to return the (real-valued) amplitude of the MPS

    Returns
    -------
    mps_freqs : array
        The values corresponding to spectral modulations,
        in cycles / octave or cycles / Hz depending on the
        units of fstep
    mps_times : array
        The values corresponding to temporal modulations, in Hz
    amps : array
        The MPS of the input strf

    """
    # Convert to frequency space and take amplitude
    nfreqs, nlags = strf.shape
    mps = np.fft.fft2(strf)
    if return_amp is True:
        mps = np.real(mps * np.conj(mps))

    # Labels for axes
    labels_freq = np.fft.fftfreq(nfreqs, fstep)
    labels_time = np.fft.fftfreq(nlags, tstep)

    # Shift to 0 center
    mps = np.fft.fftshift(mps)
    labels_freq = np.fft.fftshift(labels_freq)
    labels_time = np.fft.fftshift(labels_time)

    if return_top_half is True:
        msk_upper = labels_freq < 0
        mps = mps[msk_upper, :]
        labels_freq = labels_freq[msk_upper]
    return labels_freq, labels_time, mps

def imps(mps, out_shape=None):
    """Invert an MPS back to a spectrogram.

    Parameters
    ----------
    mps : array
        The MPS to be inverted with a 2D IFFT
    out_shape : array
        The dimensions of the output spectrogram

    Returns
    -------
    ispec : array, shape == out_shape
        The inverted spectrogram
    """
    mps = np.fft.ifftshift(mps)
    spec_i = np.real(np.fft.irfft2(mps, s=out_shape))
    return spec_i

def filter_mps(mps, mpsfreqs, mpstime, flim, tlim):
    """Filter out frequencies / times (in cycles) of the MPS.

    Parameters
    ----------
    mps : array, shape (n_freqs, n_times)
        The modulation power spectrum of a spectrogram / STRF.
    mpsfreqs : array, shape (n_freqs,)
        The values on the frequency (y) axis of the MPS
    mpstimes : array, shape (n_times,)
        The values on the time (x) axis of the MPS
    flim : array of floats | None, shape (2,)
        The minimum/maximum value on the frequency axis to keep
    tlim : array of floats | None, shape (2,)
        The minimum/maximum value on the time axis to keep

    Returns
    -------
    mps : array, shape (n_freqs, n_times)
        The MPS with various regions zeroed out
    msk_mps : array, shape (n_freqs, n_times)
        A boolean mask showing the frequencies kept
    """
    mps = mps.copy()
    msk_freq, msk_time = [np.zeros_like(mps, dtype=bool)
                      for _ in range(2)]
    use_times = mne.utils._time_mask(mpstime, *tlim)
    msk_time[:, use_times] = True

    use_freqs = mne.utils._time_mask(mpsfreqs, *flim)
    msk_freq[use_freqs, :] = True

    msk_mps = msk_time * msk_freq

    # Create random phases for the filtered parts
    msk_remove = ~msk_mps
    nphase = np.sum(msk_remove)
    phase_rand = 2 * np.pi * (np.random.rand(nphase) - .5)
    phase_rand = np.array([np.complex(0, i) for i in phase_rand])
    phase_rand = np.exp(phase_rand)

    # Now convert the masked values to 0 amplitude and rand phase
    mps[msk_remove] = 0 * phase_rand
    return mps, msk_mps


def invert_real_spectrogram(spec, win_size,
                            n_iter=20, out_length=None, tstep=None):
    """Invert a real-valued spectrogram.

    Uses the griffith/lim algorithm, a method for iteratively building a 1-d
    waveform from a spectrogram with no phase information.

    Parameters
    ----------
    spec : array, shape(n_frequencies, n_times)
        The spectrogram to be inverted. It should only contain real numbers.
        If you have a complex spectrogram, then this is more easily
        inverted with `invert_spectrogram`.
    win_size : float
        The size of the windowing function to convert back to time
    out_length : int
        The desired length of the output sound
    tstep : int
        The step between windows, in samples

    Returns
    -------
    isnd : array
        The 1-d waveform of sound corresponding to the inverted spectrogram.
        Length will depend on tstep and win_size, or out_length if given.
    spec_updated : array, shape (n_frequencies, n_times)
        The spectrogram from the last iteration of the algorithm.
    """
    # Initial inverted estimation
    tstep = win_size / 2 if tstep is None else tstep
    isnd = mne.time_frequency.istft(spec, tstep=tstep, Tx=out_length)

    for i in range(n_iter):
        # Calculate the spectrogram of the estimate
        spec_est = mne.time_frequency.stft(isnd, win_size, tstep=tstep)

        # Calculate the difference between the estimated/original spectrogram
        diff = (np.abs(spec) - np.abs(spec_est))**2
        mag = np.abs(spec)**2
        error = diff.sum() / mag.sum()
        print("Error after iteration {0}: {1}".format(i+1, error))

        # Take the angle for our estimated spectrogram and update before repeat
        spec_angle = np.angle(spec_est)
        spec_updated = spec * np.exp(1j*spec_angle)
        isnd = mne.time_frequency.istft(spec_updated, tstep=tstep, Tx=out_length)
    return isnd, spec_updated
