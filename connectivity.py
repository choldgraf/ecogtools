import numpy as np
from mne.filter import band_pass_filter
from mne.utils import _time_mask
from mne.parallel import parallel_func
from mne.time_frequency import cwt_morlet
from mne.preprocessing import peak_finder
from mne.utils import ProgressBar
from mne.baseline import rescale

# Supported PAC functions
_pac_funcs = ['plv', 'glm', 'mi_tort', 'mi_canolty', 'ozkurt', 'otc']
# Calculate the phase of the amplitude signal for these PAC funcs
_hi_phase_funcs = ['plv']


def phase_amplitude_coupling(inst, f_phase, f_amp, ixs, pac_func='ozkurt',
                             ev=None, ev_grouping=None, tmin=None, tmax=None,
                             baseline=None, baseline_kind='mean',
                             scale_amp_func=None, use_times=None, npad='auto',
                             return_data=False, concat_epochs=True, n_jobs=1,
                             verbose=None):
    """ Compute phase-amplitude coupling between pairs of signals using pacpy.

    Parameters
    ----------
    inst : an instance of Raw or Epochs
        The data used to calculate PAC
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amp : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac
    ev : array-like, shape (n_events,) | None
        Indices for events. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided
    ev_grouping : array-like, shape (n_events,) | None
        Calculate PAC in each group separately, the output will then be of
        length unique(ev)
    tmin : float | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index.
    tmax : float | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index.
    baseline : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the amplitude baseline. If None, no baseline is applied.
    baseline_kind : str
        What kind of baseline to use. See mne.baseline.rescale for options.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., skl.preprocessing.scale
    use_times : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the PAC analysis. If None, the whole window (tmin to tmax) is used.
    npad : int | 'auto'
        The amount to pad each signal by before calculating phase/amplitude if
        the input signal is type Raw. If 'auto' the signal will be padded to
        the next power of 2 in length.
    return_data : bool
        If True, return the phase and amplitude data along with the PAC values.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs, [n_events])
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs.

    References
    ----------
    [1] This function uses the PacPy modulte developed by the Voytek lab.
        https://github.com/voytekresearch/pacpy
    """
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if not isinstance(inst, (_BaseEpochs, _BaseRaw)):
        raise ValueError('Must supply either Epochs or Raw')

    sfreq = inst.info['sfreq']
    time_mask = _time_mask(inst.times, tmin, tmax)
    if isinstance(inst, _BaseRaw):
        if ev is None:
            start, stop = np.where(time_mask)[0][[0, -1]]
            data = inst[:, start:(stop + 1)][0]
        else:
            # In this case tmin/tmax are for creating epochs later
            data = inst[:, :][0]
    else:
        raise ValueError('Input must be of type Raw')
    pac = _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                                    pac_func=pac_func, ev=ev,
                                    ev_grouping=ev_grouping,
                                    tmin=tmin, tmax=tmax, baseline=baseline,
                                    baseline_kind=baseline_kind,
                                    scale_amp_func=scale_amp_func,
                                    use_times=use_times, npad=npad,
                                    return_data=return_data,
                                    concat_epochs=concat_epochs,
                                    n_jobs=n_jobs, verbose=verbose)
    # Collect the data properly
    if return_data is True:
        pac, data_ph, data_am = pac
        return pac, data_ph, data_am
    else:
        return pac


def _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='plv', ev=None, ev_grouping=None,
                              tmin=None, tmax=None,
                              baseline=None, baseline_kind='mean',
                              scale_amp_func=None, use_times=None, npad='auto',
                              return_data=False, concat_epochs=True, n_jobs=1,
                              verbose=None):
    """ Compute phase-amplitude coupling using pacpy.

    Parameters
    ----------
    data : array, shape ([n_epochs], n_channels, n_times)
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amp : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac
    ev : array-like, shape (n_events,) | None
        Indices for events. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided
    ev_grouping : array-like, shape (n_events,) | None
        Calculate PAC in each group separately, the output will then be of
        length unique(ev)
    tmin : float | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index.
    tmax : float | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index.
    baseline : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the amplitude baseline. If None, no baseline is applied.
    baseline_kind : str
        What kind of baseline to use. See mne.baseline.rescale for options.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., skl.preprocessing.scale
    use_times : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the PAC analysis. If None, the whole window (tmin to tmax) is used.
    npad : int | 'auto'
        The amount to pad each signal by before calculating phase/amplitude if
        the input signal is type Raw. If 'auto' the signal will be padded to
        the next power of 2 in length.
    return_data : bool
        If True, return the phase and amplitude data along with the PAC values.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs, [n_events])
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs.
    """
    from pacpy import pac as ppac
    if pac_func not in _pac_funcs:
        raise ValueError("PAC function {0} is not supported".format(pac_func))
    func = getattr(ppac, pac_func)
    ixs = np.array(ixs, ndmin=2)

    if data.ndim != 2:
        raise ValueError('Data must be shape (n_channels, n_times)')
    if ixs.shape[1] != 2:
        raise ValueError('Indices must have have a 2nd dimension of length 2')
    if len(f_phase) != 2 or len(f_amp) != 2:
        raise ValueError('Frequencies must be specified w/ a low/hi tuple')

    print('Pre-filtering data and extracting phase/amplitude...')
    hi_phase = pac_func in _hi_phase_funcs
    data_ph, data_am, ix_map_ph, ix_map_am = _pre_filter_ph_am(
        data, sfreq, ixs, f_phase, f_amp, npad=npad, hi_phase=hi_phase,
        scale_amp_func=scale_amp_func)
    ixs_new = [(ix_map_ph[i], ix_map_am[j]) for i, j in ixs]

    if ev is not None:
        use_times = [tmin, tmax] if use_times is None else use_times
        ev_grouping = np.ones_like(ev) if ev_grouping is None else ev_grouping
        data_ph, times, msk_ev = _array_raw_to_epochs(
            data_ph, sfreq, ev, tmin, tmax)
        data_am, times, msk_ev = _array_raw_to_epochs(
            data_am, sfreq, ev, tmin, tmax)

        # In case we cut off any events
        ev, ev_grouping = [i[msk_ev] for i in [ev, ev_grouping]]

        # Baselining before returning
        rescale(data_am, times, baseline, baseline_kind, copy=False)
        msk_time = _time_mask(times, *use_times)
        data_am, data_ph = [i[..., msk_time] for i in [data_am, data_ph]]

        # Stack epochs to a single trace if specified
        if concat_epochs is True:
            ev_unique = np.unique(ev_grouping)
            concat_data = []
            for i_ev in ev_unique:
                msk_events = ev_grouping == i_ev
                concat_data.append([np.hstack(i[msk_events])
                                    for i in [data_am, data_ph]])
            data_am, data_ph = zip(*concat_data)
    else:
        data_ph, data_am = [data_ph], [data_am]

    n_ep = len(data_ph)
    pac = np.zeros([n_ep, len(ixs_new)])
    pbar = ProgressBar(n_ep)
    for iep, (ep_ph, ep_am) in enumerate(zip(data_ph, data_am)):
        for iix, (i_ix_ph, i_ix_am) in enumerate(ixs_new):
            # f_phase and f_amp won't be used in this case
            pac[iep, iix] = func(ep_ph[i_ix_ph], ep_am[i_ix_am],
                                 f_phase, f_amp, filterfn=False)
        pbar.update_with_increment_value(1)
    if return_data:
        return pac, data_ph, data_am
    else:
        return pac


def _pre_filter_ph_am(data, sfreq, ixs, f_ph, f_am, filterfn=None,
                      npad=None, hi_phase=False, scale_amp_func=None,
                      kws_filt=None):
    """Filter for phase/amp only once for each channel."""
    from pacpy.pac import _range_sanity
    from scipy.signal import hilbert
    filterfn = band_pass_filter if filterfn is None else filterfn
    kws_filt = dict() if kws_filt is None else kws_filt
    _range_sanity(f_ph, f_am)
    ix_ph = np.atleast_1d(np.unique(ixs[:, 0]))
    ix_am = np.atleast_1d(np.unique(ixs[:, 1]))

    # For padding if input is shape of raw
    n_times = data.shape[-1]
    if npad == 'auto':
        next_pow_2 = int(np.ceil(np.log2(n_times)))
        npad = 2**next_pow_2 - n_times
    elif not isinstance(npad, int):
        raise ValueError('npad must be "auto" or a positive integer')

    data_ph = np.hstack([data[ix_ph, :], np.zeros([len(ix_ph), npad])])
    data_ph = filterfn(data_ph, sfreq, *f_ph, **kws_filt)
    data_ph = np.angle(hilbert(data_ph))[..., :n_times]
    ix_map_ph = {ix: i for i, ix in enumerate(ix_ph)}

    data_am = np.hstack([data[ix_am], np.zeros([len(ix_am), npad])])
    data_am = filterfn(data_am, sfreq, *f_am, **kws_filt)
    data_am = np.abs(hilbert(data_am))
    if hi_phase is True:
        data_am = filterfn(data_am, sfreq, *f_ph, **kws_filt)
        data_am = np.angle(hilbert(data_am))
    data_am = data_am[..., :n_times]
    ix_map_am = {ix: i for i, ix in enumerate(ix_am)}

    if scale_amp_func is not None:
        data_am = scale_amp_func(data_am, axis=-1)
    return data_ph, data_am, ix_map_ph, ix_map_am


def _filter_ph_am(xph, xam, f_ph, f_am, sfreq, filterfn=None, kws_filt=None):
    """Aux function for phase/amplitude filtering for one pair of channels"""
    from pacpy.pac import _range_sanity
    from scipy.signal import hilbert
    filterfn = band_pass_filter if filterfn is None else filterfn
    kws_filt = {} if kws_filt is None else kws_filt

    # Filter the two signals + hilbert/phase
    _range_sanity(f_ph, f_am)
    xph = filterfn(xph, sfreq, *f_ph)
    xam = filterfn(xam, sfreq, *f_am)

    xph = np.angle(hilbert(xph))
    xam = np.abs(hilbert(xam))
    return xph, xam


def _array_raw_to_epochs(x, sfreq, ev, tmin, tmax):
    """Aux function to create epochs from a 2D array"""
    if ev.ndim != 1:
        raise ValueError('ev must be 1D')
    if ev.dtype != int:
        raise ValueError('ev must be of dtype int')

    # Check that events won't be cut off
    n_times = x.shape[-1]
    min_ix = 0 - sfreq * tmin
    max_ix = n_times - sfreq * tmax
    msk_keep = np.logical_and(ev > min_ix, ev < max_ix)

    if not all(msk_keep):
        print('Some events will be cut off!')
        ev = ev[msk_keep]

    # Pull events from the raw data
    epochs = []
    for ix in ev:
        ix_min, ix_max = [ix + int(i_tlim * sfreq)
                          for i_tlim in [tmin, tmax]]
        epochs.append(x[np.newaxis, :, ix_min:ix_max])
    epochs = np.concatenate(epochs, axis=0)
    times = np.arange(epochs.shape[-1]) / float(sfreq) + tmin
    return epochs, times, msk_keep


# For the viz functions
def _extract_phase_and_amp(data_phase, data_amp, sfreq, freqs_phase,
                           freqs_amp, scale=True):
    """Extract the phase and amplitude of two signals for PAC viz.
    data should be shape (n_epochs, n_times)"""
    from sklearn.preprocessing import scale

    # Morlet transform to get complex representation
    band_ph = cwt_morlet(data_phase, sfreq, freqs_phase)
    band_amp = cwt_morlet(data_amp, sfreq, freqs_amp)

    # Calculate the phase/amplitude of relevant signals across epochs
    band_ph_stacked = np.hstack(np.real(band_ph))
    angle_ph = np.hstack(np.angle(band_ph))
    amp = np.hstack(np.abs(band_amp) ** 2)

    # Scale the amplitude for viz so low freqs don't dominate highs
    if scale is True:
        amp = scale(amp, axis=1)
    return angle_ph, band_ph_stacked, amp


def _pull_data(inst, ix_ph, ix_amp, ev=None, tmin=None, tmax=None):
    """Pull data from either Base or Epochs instances"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if isinstance(inst, _BaseEpochs):
        data_ph = inst._data[:, ix_ph, :]
        data_amp = inst._data[:, ix_amp, :]
    elif isinstance(inst, _BaseRaw):
        data = inst[[ix_ph, ix_amp], :][0].squeeze()
        data_ph, data_amp = [i[np.newaxis, ...] for i in data]
    return data_ph, data_amp


def phase_locked_amplitude(inst, freqs_phase, freqs_amp,
                           ix_ph, ix_amp, tmin=-.5, tmax=.5,
                           mask_times=None):
    """Calculate the average amplitude of a signal at a phase of another.

    Parameters
    ----------
    inst : instance of mne.Epochs or mne.io.Raw
        The data to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    tmin : float
        The time to include before each phase peak
    tmax : float
        The time to include after each phase peak
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        If inst is an instance of Raw, this will only include times contained
        in mask_times.

    Returns
    -------
    data_amp : np.array
        The mean amplitude values for the frequencies specified in freqs_amp,
        time-locked to peaks of the low-frequency phase.
    data_phase : np.array
        The mean low-frequency signal, phase-locked to low-frequency phase
        peaks.
    times : np.array
        The times before / after each phase peak.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_amp = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_amp, sfreq, freqs_phase, freqs_amp)

    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    band_ph = band_ph.mean(0)

    # Find peaks in the phase for time-locking
    phase_peaks, vals = peak_finder.peak_finder(angle_ph)
    ixmin, ixmax = [t * sfreq for t in [tmin, tmax]]
    # Remove peaks w/o buffer
    phase_peaks = phase_peaks[(phase_peaks > np.abs(ixmin)) *
                              (phase_peaks < len(angle_ph) - ixmax)]

    if mask_times is not None:
        # Set datapoints outside out times to nan so we can drop later
        if len(mask_times) != angle_ph.shape[-1]:
            raise ValueError('mask_times must be == in length to data')
        band_ph[..., ~mask_times] = np.nan

    data_phase, times, msk_window = _array_raw_to_epochs(
        band_ph[np.newaxis, :], sfreq, phase_peaks, tmin, tmax)
    data_amp, times, msk_window = _array_raw_to_epochs(
        amp, sfreq, phase_peaks, tmin, tmax)
    data_phase, data_amp = [i.squeeze() for i in data_phase, data_amp]

    # Drop any peak events where there was a nan
    keep_rows = np.where(~np.isnan(data_phase).any(-1))[0]
    data_phase, data_amp = [i[keep_rows, ...] for i in [data_phase, data_amp]]

    # Average across phase peak events
    data_amp = data_amp.mean(0)
    data_phase = data_phase.mean(0)
    return data_amp, data_phase, times


def phase_binned_amplitude(inst, freqs_phase, freqs_amp,
                           ix_ph, ix_amp, n_bins=20, mask_times=None):
    """Calculate amplitude of one signal in sub-ranges of phase for another.

    Parameters
    ----------
    inst : instance of mne.Epochs or mne.io.Raw
        The data to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size (2 * np.pi) / n_bins.
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        If inst is an instance of Raw, this will only include times contained
        in mask_times.

    Returns
    -------
    amp_binned : np.array, shape (n_bins,)
        The mean amplitude of freqs_amp at each phase bin
    bins_phase : np.array, shape (n_bins+1)
        The bins used in the calculation. There is one extra bin because
        bins represent the left/right edges of each bin, not the center value.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_amp = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_amp, sfreq, freqs_phase, freqs_amp)
    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    if mask_times is not None:
        # Only keep times we want
        if len(mask_times) != amp.shape[-1]:
            raise ValueError('mask_times must be == in length to data')
        angle_ph, band_ph, amp = [i[..., mask_times]
                                  for i in [angle_ph, band_ph, amp]]

    # Bin our phases and extract amplitudes based on bins
    bins_phase = np.linspace(-np.pi, np.pi, n_bins)
    bins_phase_ixs = np.digitize(angle_ph, bins_phase)
    unique_bins = np.unique(bins_phase_ixs)
    amp_binned = [np.mean(amp[:, bins_phase_ixs == i], 1)
                  for i in unique_bins]
    amp_binned = np.vstack(amp_binned).mean(1)

    return amp_binned, bins_phase
