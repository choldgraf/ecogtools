import mne
from glob import glob
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import h5io

__all__ = ['EpochTFR', 'tfr_epochs', 'extract_amplitude']


class EpochTFR(object):
    """
    Create an EpochTFR object for single-trial TFR representation.

    This allows you to store TFR information without averaging across
    trials. Be careful with memory usage, as it can get quite large.
    Consider decimating before this step.

    Parameters
    ----------
    data : ndarray, shape(n_epochs, n_chans, n_freqs, n_times)
        The TFR data you wish to construct.
    freqs : ndarray, shape(n_freqs)
        The frequency labels during TFR construction
    times : ndarray, shape(n_times)
        The time labels during epochs creation
    events : ndarray, shape (n_epochs, 3)
        An MNE events array, see documentation for details
    event_id : dict
        A mapping from event names to integer event IDs
    info : mne Info object
        The info object from the original Epochs instance.
        May be unnecessary but keep it just in case.
    method : string
        Information describing how this TFR data was created.
    
    Attributes
    ----------
    ch_names : list
        The names of the channels
    """
    def __init__(self, data, freqs, times, events, event_id, info, method=None):
        if data.ndim != 4:
            raise ValueError('data should be 4d. Got %d.' % data.ndim)
        n_epochs, n_channels, n_freqs, n_times = data.shape
        if n_channels != len(info['chs']):
            raise ValueError("Number of channels and data size don't match"
                             " (%d != %d)." % (n_channels, len(info['chs'])))
        if n_freqs != len(freqs):
            raise ValueError("Number of frequencies and data size don't match"
                             " (%d != %d)." % (n_freqs, len(freqs)))
        if n_times != len(times):
            raise ValueError("Number of times and data size don't match"
                             " (%d != %d)." % (n_times, len(times)))
        if n_epochs != events.shape[0]:
            raise ValueError("Number of epochs and data size don't match"
                             " (%d != %d)." % (n_epochs, events.shape[0]))
        self.data = data
        self.freqs = freqs
        self.events = events
        self.event_id = event_id
        self.times = times
        self.info = info
        self.method = method

    @property
    def ch_names(self):
        return self.info['ch_names']

    def write_hdf(self, fname, **kwargs):
        """
        Write the object to an HDF file.

        This bottles up all object attributes into a dictionary,
        and sends it to an HDF file via the h5py library.

        Parameters
        ----------
        fname : str
            The file name of the hdf5 file.
        **kwargs : dict
            Arguments are passed to write_hdf5
        """
        print('Writing TFR data to {0}'.format(fname))
        write_dict = {'data': self.data, 'freqs': self.freqs,
                      'events': self.events, 'times': self.times,
                      'info': self.info, 'event_id': self.event_id}
        h5io.write_hdf5(fname, write_dict, **kwargs)
        
    def subset_freq(self, fmin, fmax):
        """
        Return the mean of a subset of frequencies as an Epochs object.
        
        Parameters
        ----------
        fmin : float
            The minimum frequency to keep.
            
        fmax : float
            The maximum frequency to keep.
        """
        mask_freq = (self.freqs > fmin) * (self.freqs < fmax)
        data_freq = self.data[:, :, mask_freq, :].mean(2)
        epoch_info = self.info.copy()
        epoch_info['description'] = "{'kind': 'TFRFreqSubset', 'freqs': '{0}-{1}'.format(fmin, fmax)}"
        return mne.EpochsArray(data_freq, epoch_info, self.events,
                                 tmin=self.times.min(), event_id=self.event_id)

    def subset_epoch(self, epoch_ixs):
        """
        Return the mean of a subset of epochs.
        
        Parameters
        ----------
        epoch_ixs : list, ndarray
            The epoch indices to keep

        Returns
        ----------
        obj : AverageTFR
            Instance of AverageTFR with epochs averaged.
        """

        data_epoch = self.data[epoch_ixs, ...].squeeze().mean(0)
        epoch_info = self.info.copy()
        return mne.time_frequency.AverageTFR(epoch_info, data_epoch, self.times,
                                             self.freqs, len(epoch_ixs), comment='TFREpochSubset',
                                             method=self.method)

    def crop(self, tmin=None, tmax=None, copy=False):
        """Crop data to a given time interval

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        copy : bool
            If False epochs is cropped in place.
        """
        inst = self if not copy else self.copy()
        mask_time = mne.utils._time_mask(self.times, tmin, tmax)
        inst.times = inst.times[mask_time]
        inst.data = inst.data[..., mask_time]
        return inst

    def iter(self, epochs=None, picks=None):
        """Iterate a subset of epochs / channels.

        Parameters
        ----------
        epochs : list | None
            A list of epochs to include in each iteration.
        picks : list | None
            A list of picks to iterate over. Each iteration
            will contain one electrode.

        Returns
        -------
        ep_iter : generator
            Each iteration will be shape(n_epochs, n_freqs, n_times)
        """
        ep_iter = (self.data[epochs, i, ...] for i in picks)
        return ep_iter

    def __repr__(self):
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ', epochs: {0}'.format(self.events.shape[0])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ', channels : %d' % self.data.shape[0]
        return "<EpochTFR  |  %s>" % s

    @staticmethod
    def from_hdf(fname):
        """
        Read an EpochsTFR from an HDF5 file.
        
        This expects an HDF5 file that was created with
        write_hdf5. It auto-populates a new EpochsTFR object.
        Be careful with memory consumption!
        
        Parameters
        ----------
        fname : str
            The path to the HDF5 file you wish to import.
            
        Returns
        -------
        etfr : EpochsTFR
            The EpochsTFR object contained in the HDF5 file.
        """
        params = h5io.read_hdf5(fname)
        etfr = EpochTFR(**params)
        return etfr


def tfr_epochs(epoch, freqs_tfr, n_decim=10, n_cycles=5):
    """Extract a TFR representation from epochs w/o averaging.

    This will extract the TFR for the given frequencies, returning
    an EpochsTFR object with data shape (n_epochs, n_chans, n_freqs, n _times).
    TFR is performed with a morlet wavelet.

    Parameters
    ----------
    epoch : mne.Epochs instance
        The epochs for which to extract the TFR
    freqs_tfr : array-like
        The frequencies to extract w/ wavelets
    n_decim : int
        The factor to decimate the time-dimension. 1=no decimation
    n_cycles : int
        The number of cycles for the wavelets
    """
    # Preallocate space
    new_times = epoch.times[::n_decim]
    new_info = epoch.info.copy()
    new_info['sfreq'] /= n_decim
    new_epoch = epoch.copy()
    n_epoch = len(new_epoch)
    out_shape = (n_epoch, len(epoch.ch_names),
                 freqs_tfr.shape[0], len(new_times))
    tfr = np.zeros(out_shape, dtype=np.float32)
    print('Preallocating space\nshape: {0}\nmbytes:{1}'.format(
        out_shape, tfr.nbytes/1000000))

    # Do TFR decomp
    print('Extracting TFR information...')
    for i in tqdm(range(n_epoch)):
        iepoch = new_epoch[i]
        idata = iepoch._data.squeeze()
        itfr = mne.time_frequency.cwt_morlet(idata, iepoch.info['sfreq'],
                                             freqs_tfr, n_cycles=n_cycles)
        itfr = itfr[:, :, ::n_decim]
        tfr[i] = np.abs(itfr)

    etfr = EpochTFR(tfr, freqs_tfr, new_times, new_epoch.events,
                    new_epoch.event_id, new_info)
    return etfr


def extract_amplitude(inst, freqs, normalize=False, new_len=None,
                      picks=None, copy=True, n_jobs=1, ):
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
    normalize : bool
        Whether to normalize each frequency amplitude by its mean before
        averaging. This can be helpful if some frequencies have a much higher
        base amplitude than others.
    new_len : int | None
        The length of data to use in the Hilbert transform. The data will be
        cut to last dimension of this size.
    picks : array | None
        The channels to use for extraction

    Returns
    -------
    inst : mne instance, same type as input 'inst'
        The MNE instance with channels replaced with their time-varying
        amplitude for the supplied frequency range.
    """

    # Data checks
    new_len = inst.n_times if new_len is None else new_len
    freqs = np.atleast_2d(freqs)
    if freqs.shape[-1] != 2:
        raise ValueError('freqs must be shape (n_fbands, 2)')
    picks = range(len(inst.ch_names)) if picks is None else picks

    # Filter for HG and extract amplitude
    bands = np.zeros([freqs.shape[0], len(inst.ch_names), inst.n_times])
    for i, (fmin, fmax) in enumerate(freqs):
        # Extract a range of frequency bands for averaging later
        inst_band = inst.copy()
        inst_band.filter(fmin, fmax, n_jobs=n_jobs)
        inst_band.apply_hilbert(picks, envelope=True,
                                n_jobs=n_jobs, n_fft=new_len)

        if normalize is True:
            # Scale frequency band so that the ratios of all are the same
            inst_band_mn = inst_band._data.mean(1)[:, np.newaxis]
            inst_band._data /= inst_band_mn
        bands[i] = inst_band._data.copy()

    # Average across fbands
    if copy is True:
        inst = inst.copy()
    inst._data[picks, :] = bands.mean(0)
    return inst
