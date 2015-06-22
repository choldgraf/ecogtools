import mne
from glob import glob
import pandas as pd
import numpy as np
import sys
from mne.time_frequency import write_tfrs, compute_epochs_psd
from scipy.signal import decimate

__all__ = ['EpochTFR']


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
        mne._hdf5.write_hdf5(fname, write_dict, **kwargs)
        
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
        params = mne._hdf5.read_hdf5(fname)
        etfr = EpochTFR(**params)
        return etfr