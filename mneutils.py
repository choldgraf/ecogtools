"""Some helper funcs for MNE-python"""
import mne
import numpy as np
import pandas as pd
from ecogtools.utils import embed


def create_random_events(nev, ntimes, nclasses):
    """Creates a random set of nev events for each of
    nclasses, randomly interspersed in ntimes."""
    ixs = np.random.permutation(range(ntimes))
    classes = np.tile(range(nclasses), (nev, 1)).ravel()
    ixs_ev = ixs[:nev*nclasses]
    events = np.zeros([nev*nclasses, 3])
    for i, (ix, evclass) in enumerate(zip(ixs_ev, classes)):
        events[i, :] = [ix, 0, evclass]
    return events.astype(int)


def create_random_epochs(nep, nchan, ntime, sfreq,
                         nclasses=2, ch_types='eeg'):
    data = np.random.randn(nep*nclasses, nchan, ntime*sfreq)
    ev = create_random_events(nep, ntime*sfreq, nclasses)
    info = mne.create_info([str(i) for i in range(nchan)], sfreq, ch_types)
    ep = mne.epochs.EpochsArray(data, info, ev)
    return ep


def create_random_raw(nchan, ntime, sfreq, ch_types='eeg'):
    data = np.random.randn(nchan, ntime*sfreq)
    info = mne.create_info([str(i) for i in range(nchan)], sfreq, ch_types)
    raw = mne.io.RawArray(data, info)
    return raw


class Events(object):
    """Container for events metadata"""
    def __init__(self, time_start, sfreq=None, time_stop=None, meta=None):
        """Create an Events object to easily translate to MNE events array.

        Parameters
        ----------
        time_start : array, shape (n_events,)
            An array of start times (in seconds) for the events of interest.
        sfreq : None | float
            The default sampling frequency for these events.
        time_stop : None | array, shape (n_events,)
            Optionally, the stop time for each event. Must be the same length
            as time_start.
        meta : None | pandas DataFrame, shape (n_events, n_meta_columns)
            An optional metadata container for events. This is expected to have
            event type strings as values, with each row corresponding to an
            event and each column corresponding to a class of event types.

        Attributes
        ----------
        length : array, shape (n_events,)
            If time_stop is given, it is the length of each event.
        """
        self.time_start = np.array(time_start)
        if time_stop is not None:
            time_stop = np.array(time_stop)
            if time_stop.shape != time_start.shape:
                raise ValueError('start/stop mismatch')
            self.length = time_stop - self.time_start
        if meta is not None:
            if not isinstance(meta, pd.DataFrame):
                raise AssertionError('metadata must be a dataframe')
            if meta.shape[0] != len(time_start):
                raise ValueError('metadata / time_start length mismatch')
            if np.sum(meta.isnull().values) > 0:
                raise ValueError('Found NaNs in metadata')
            meta = meta.reset_index(drop=True)
        self.meta = meta
        self.sfreq = sfreq
        self.ids = np.arange(len(time_start))

    def to_mne(self, sfreq=None, meta_columns=None):
        """Convert the events to MNE shape.

        Parameters
        ----------
        sfreq : None | float
            The sampling frequency used to calculate indices
        meta_columns : None | list of strings
            The columns to use as trial types in the output array. If
            a list is given, each string in `meta_columns` must be in
            self.meta.columns. For each epoch (row) of self.meta, unique
            combinations of epoch types are tallied, and the corresponding
            strings/event_id mappings are given in einf. If None, then
            a unique string for each epoch will be generated.

        Returns
        -------
        events : array, shape (n_epochs, 3)
            An MNE events array.
        einf : dict
            The dictionary mapping of event_string: event_id for epoch types.
        """
        sfreq = self.sfreq if sfreq is None else sfreq
        if sfreq is None:
            raise ValueError('A sampling frequency must be given')
        times = self.time_start * sfreq
        zeros = np.zeros_like(times)

        if meta_columns is not None:
            iter_meta = self.meta[meta_columns]
            m_str = ['/'.join(['{0}__{1}'.format(col, val)
                               for col, val in zip(iter_meta.columns, rw)])
                     for _, rw in iter_meta.iterrows()]
            ev_id = {str(i_mstr): i + 1
                     for i, i_mstr in enumerate(np.unique(m_str))}
            events = np.array([ev_id[i_mstr] for i_mstr in m_str])
        else:
            events = np.arange(times.shape[0])
            ev_id = {'event_{0}'.format(i): i for i in self.ids}
        events = np.vstack([times, zeros, events]).T.astype(int)
        return events, ev_id
