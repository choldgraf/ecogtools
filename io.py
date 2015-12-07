import h5py

def read_strings_from_hdf_matlab_cell(cell):
    """Reads strings stored in a cell array in Matlab
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
    strs = [''.join([unichr(j) for j in f[i][:]]) for i in cell[:].ravel()]
    return strs