"""Some helper funcs for MNE-python"""
import mne
import numpy as np
import pandas as pd


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
    def __init__(self, time_start, sfreq, time_stop=None, meta=None):
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
        self.meta = meta
        self.sfreq = sfreq
        self.ids = np.arange(len(time_start))

    def to_mne(self, meta_columns=None):
        times = self.time_start * self.sfreq
        zeros = np.zeros_like(times)

        if meta_columns is not None:
            m_str = ['/'.join(rw[meta_columns].values)
                     for _, rw in self.meta.iterrows()]
            ev_id = {i_mstr: i + 1 for i, i_mstr in enumerate(np.unique(m_str))}
            events = np.array([ev_id[i_mstr] for i_mstr in m_str])
        else:
            events = np.arange(times.shape[0])
            ev_id = {'event_{0}'.format(i): i for i in self.ids}
        events = np.vstack([times, zeros, events]).T.astype(int)
        return events, ev_id
