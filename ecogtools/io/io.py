"""Functions for data IO"""
from .bci2000 import bcistream
import mne
import numpy as np


def read_strings_from_hdf_matlab_cell(cell):
    """Reads strings stored in a cell array in Matlab.
    using the -v7.3 flag.

    Matlab stores strings as 2 byte integers. If we know that the cell
    array has strings in it, we use unichr to return the string contents.

    Parameters
    ----------
    cell : h5py dataset
        The h5py dataset corresponding to the cell array of interest. It
        should simply be a list of strings.

    Returns
    -------
    strs : list
        A list of the strings contained in the h5py dataset
    """
    strs = [''.join([unichr(j) for j in f[i][:]])
            for i in cell[:].ravel()]
    return strs


class BCI2000Reader(object):
    """Read in BCI2000 data.

    Note: This only works in Python 2.X

    Read in a `.dat` file collected with bci2000. This is a
    little bit clunky, as the bci2000 code is quite old. It is
    basically a wrapper around the bci2000 functions.

    Inputs
    ------
    path : string
        The path to the .dat file to read.
    preload : bool
        Whether to preload the data. Defaults to True.

    Attributes
    ----------
    stream : instance of bcistream
        The bcistream created with `bcistream`. It contains
        the raw BCI2000 object.
    parameters : dict
        A dictionary of parameters corresponding to the data.
    parameter_definitions : dict
        A dictionary designating what parameters actually mean.
    sfreq : int | float
        The sampling frequency of the data.
    state_definitions : dict
        A dictionary designating what states actually mean.
    data : array, shape (n_channels, n_times)
        The data.
    states : array, shape (n_states, n_times)
        The value of each state during the experiment.
    """
    def __init__(self, path, preload=True):
        self.stream = stream = bcistream(path)
        self.parameters = stream.params
        self.parameter_definitions = stream.paramdefs
        self.sfreq = stream.samplingrate()
        self.state_definitions = stream.statedefs
        self.preload = preload
        self.data = None

        self.ch_names = self.parameters['ChannelNames']
        if preload is True:
            self.load_data()

    def load_data(self):
        """Load the data in the stream object."""
        self.data, self.states = self.stream.read('all')
        self.data = np.array(self.data)

    def to_mne(self, states=True):
        """Convert data into an MNE Raw object.

        Parameters
        ----------
        states : bool
            Whether to include state channels in the output.

        Returns
        -------
        raw : instance of MNE Raw
            The data in MNE Raw format.
        """
        if len(self.ch_names) == 0:
            ch_names = ['ch_{}'.format(ii)
                        for ii in range(self.data.shape[0])]
        ch_types = ['eeg'] * len(ch_names)
        if states is True:
            state_names = ['state_{}'.format(ii)
                           for ii in range(self.states.shape[0])]
            state_types = ['misc'] * len(state_names)
            data = np.vstack([self.data, self.states])
        else:
            state_names = state_types = []
            data = self.data
        info = mne.create_info(ch_names + state_names, self.sfreq,
                               ch_types + state_types)
        raw = mne.io.RawArray(data, info)
        return raw
