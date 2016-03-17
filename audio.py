"""A collection of functions for auditory processing and analysis."""
import numpy as np
import pandas as pd
from brian import Hz, kHz
from brian import hears


def create_torc(fmin, fmax, n_freqs, sfreq, duration=1., rip_spec_freq=2,
                rip_temp_freq=2, mod_depth=.9, rip_phase_offset=0,
                time_buffer=.5, combine_waves=True):
    """Create a torc stimulus.

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


def spectrogram_audio(audio, n_bands=32, sfreq=44100,
                      sig_fac=.1, compression='log',
                      low_p_cut=None, lin=True, n_jobs=3,
                      filt_kind='nsl', freq_kind='erb',
                      Flo=170, Fhi=7000, amp='atonce'):
    ''' Extracts a (roughly) auditory system spectrogram.
    This is loosely based on the NSL toolbox. Note that many
    of these steps can be controlled with various flags
    defined above.

    Here are the steps it takes:
        1. Filter the sound with a frequencies that are erb log-spaced
        2. Extract the analytic amplitude of the sound
        3. Compression with a sigmoid
        4. Low-pass filtering this amplitude
        5. First-order derivative across frequencies (basically just
            taking the diff of successive frequencies)
        6. Half-wave rectification

    Parameters
    ----------
    audio : array, shape (n_times,)
        The input sound.
    n_bands : int, default=32
        The number of frequency bands in our filter
    sig_fac : float
        The sigmoidal compression factor. See `compress_signal` for usage
    lin : bool
        Whether to include the first order derivative
        AKA the lateral inhibitory network
    low_p_cut : int | None
        The cutoff for the lowpass filter, or None for no filter
    filt_kind : one of ['drnl', 'nsl']
        How to extract the spectrogram. Options mean:
        drnl : a self-contained cochlea model,
               so we don't add any extra processing afterward. However,
               it seems to be unstable for high F (>5000). Look into brian.hears
               for more documentation on this.
        nsl : an implementation of the wav2aud function in the NSL toolbox.
              It is implemented with brian.hears
    freq_kind : string ['erb', 'log']
        What frequency spacing to use
    amp : string ['online', 'atonce']
        Do we calculate the envelope of the signal online or at once?

    OUTPUTS
    --------
    out     : DataFrame, time x features
        The extracted spectrogram.

    '''
    # Auditory filterbank + amplitude extraction
    print('Running filterbank with {0} filters'.format(n_bands))
    csfreq = create_center_frequencies(Flo, Fhi, n_bands, kind=freq_kind)

    if filt_kind == 'drnl':
        sfreq = float(sfreq)*Hz
        snd = hears.Sound(audio, samplerate=sfreq)
        spec = hears.DRNL(snd, cfs, type='human').process()
        return spec, cfs
    elif filt_kind == 'nsl':
        spec = spectrogram_nsl(audio, sfreq, cfs)
        return spec, cfs


def spectrogram_nsl(sig, sfreq, cfs, comp_kind='exp', comp_fac=3):
    '''Extract a cochlear / mid-brain spectrogram.

    Implements a version of the "wav2aud" function in the NSL toolbox.
    Uses Brian hears to chain most of the computations to be done online.

    This is effectively what it does:
        1. Gammatone filterbank at provided cfs (erbspace recommended)
        2. Half-wave rectification
        3. Low-pass filtering at 2Khz
        4. First-order derivative across frequencies (basically just
            taking the diff of successive frequencies to sharpen output)
        5. Half-wave rectification #2
        6. An exponentially-decaying average, with time constant chosen
            to be similar to that reported in the NSL toolbox (8ms)

    Parameters
    ----------
    sig : numpy array, shape (n_times,)
        The auditory waveform
    sfreq : int
        The sampling frequency of the sound waveform
    cfs : array, shape (n_freqs,)
        The center frequencies to be extracted
    comp_kind : string
        The kind of compression to use. See `compress_signal`
    comp_fac : int
        The compression factor to pass to `compress_signal`.

    Returns
    -------
    out : array, shape (n_frequencies, n_times)
        The spectrogram representation of the input sound.
    '''
    sfreq = float(sfreq)*Hz
    snd = hears.Sound(sig, samplerate=sfreq)

    # ---- Cochlear model
    print('Pulling frequencies with cochlear model')
    snd_filt = hears.Gammatone(snd, cfs)

    # ---- Hair cell stages
    # Halfwave Rectify
    print('Half-wave rectification')
    clp = lambda x: np.clip(x, 0, np.inf)
    snd_hwr = hears.FunctionFilterbank(snd_filt, clp)

    # Non-linear compression
    print('Non-linear compression and low-pass filter')
    comp = lambda x: compress_signal(x, comp_kind, comp_fac)
    snd_cmp = hears.FunctionFilterbank(snd_hwr, comp)

    # Lowpass filter
    snd_lpf = hears.LowPass(snd_cmp, 2000)

    # ---- Lateral inhibitory network
    print('Lateral inhibitory network')
    rands = lambda x: sigp.roll_and_subtract(x, hwr=True)
    snd_lin = hears.FunctionFilterbank(snd_lpf, rands)

    # Initial processing
    out = snd_lin.process()

    # Time integration.
    print('leaky integration')
    for i in range(out.shape[1]):
        out[:, i] = sigp.leaky_integrate(out[:, i], time_const=8,
                                         sfreq=float(sfreq))
    return out


def leaky_integrate(arr, time_const=8, sfreq=1000):
    '''
    Performs a leaky integration on the array "arr", with a time constant
    equal to the number of timepoints until the signal drops by 67%.

    sfreq is the sampling rate of the signal, and time_const is in the units
    of this sampling rate. AKA, if sfreq is 1000 and time_const is 8, then the
    time constant is 8ms.

    Parameters
    ----------
    arr : array, shape (n_times,)
        The array to integrate over time
    time_const : int
        The time constant in milliseconds.
    sfreq : int
        The sampling frequency of arr

    Returns
    -------
    out : array, shape (n_times)
        The integrated signal.
    '''
    time = np.arange(sfreq)

    # Convert time constant from ms to whatever Fs is
    time_const = float(sfreq / 1000.) * time_const
    weights = np.exp(-(time / time_const)) * np.ones_like(time)
    out = fftconvolve(arr, weights)[:-sfreq + 1]
    return out


def create_center_frequencies(stt=180, stp=7000, n_bands=32, kind='log'):
    '''
    Define center frequencies for spectrograms.

    Generally this is for auditory spectrogram extraction. Most auditory
    analysis uses 180 - 7000 Hz, so for now those
    are the defaults.

    Parameters
    ----------
    stt : float | int
        The starting frequency
    stp : float | int
        The end frequency
    n_bands : int
        The number of bands to calculate
    kind : 'log' | 'erb'
        Whether to use log or erb spacing

    Returns
    -------
    freqs : array, shape (n_frequencies,)
        An array of center frequencies.
    '''
    if kind == 'log':
        freqs = np.logspace(np.log10(stt), np.log10(stp), n_bands).astype(int)
    elif kind == 'erb':
        freqs = hears.erbspace(stt * Hz, stp * Hz, n_bands)
    else:
        print("I don't know what kind of spacing that is")
    return freqs
