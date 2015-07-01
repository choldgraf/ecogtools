"""Cross validation extras."""

import numpy as np
from sklearn import cross_validation as cval


class BlockShuffleSplit(object):

    """
    Iterate over blocks of indices.

    Cross-validation iterator to shuffle indices based off of labels.
    These labels can either be given explicitly, or generated in blocks
    of length n_in_block. The train/test split is based off of the
    number of block labels, so that no indices from one label will ever
    make into both the train and test splits.

    Parameters
    ----------
    n : int
        The length of the indices you wish to generate.
    n_iter : int
        The number of iterations for this CV iterator
    n_in_block : int
        If automatically generating blocks of labels, how many indices
        do we want per block.
    block_indices : array, same length as n
        Explicitly define block labels. This is useful if you wish to
        group data points into, for example, trials or trial types.
        Nullifies
    test_size : float (between 0 and 1)
        The proportion of block labels in the test set of each CV.
        If block labels do not separate into an integer number in each
        split, then the number in the test set is rounded down (so the
        test set may have more labels than you thought.)
    """

    def __init__(self, block_indices=None, n=None, n_iter=10,
                 n_in_block=10, test_size=0.1, random_state=None):

        if block_indices is None:
            if n is None or n_in_block is None:
                raise AssertionError('Supply block_indices, or n + n_in_block')
            n_blocks = np.ceil(float(n) / n_in_block)
            self.block_list = np.arange(n_blocks)
            self.block_indices = np.hstack([[i]*n_in_block
                                            for i in self.block_list])[:n]
        else:
            print('Using pre-supplied block indices')
            self.block_indices = block_indices
            self.block_list = np.unique(block_indices)
            n = block_indices.shape[0]

        self.n = n
        self.n_iter = n_iter
        self.n_in_block = n_in_block
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.random_stats = random_state
        self.indices = np.arange(n)
        self.n_blocks = len(np.unique(self.block_indices))
        if self.block_indices.shape[0] != self.n:
            raise AssertionError('Unequal n and block_indices')
        print('Using {0} blocks'.format(self.n_blocks))

    def __len__(self):
        """Return the number of iterations to use."""
        return self.n_iter

    def __iter__(self):
        """Iterate over blocks of indices."""
        for _ in range(self.n_iter):
            ixs = shuffle_block(self.indices, n=self.n_in_block,
                                block_indices=self.block_indices)
            cut = int(np.floor(len(self.block_list) * self.train_size))
            train_blocks = self.block_list[:cut]
            test_blocks = self.block_list[cut:]
            train_i = ixs[np.array([i in train_blocks
                                    for i in self.block_indices])]
            test_i = ixs[np.array([i in test_blocks
                                   for i in self.block_indices])]
            yield train_i, test_i


def shuffle_block(arr, n=10, block_indices=None):
    '''Shuffles the values in a in a blockwise fashion. This lets you shuffle
    the values, but keeping nearby ones together.

    Parameters
    ----------
    arr : np.array
        The array you wish to shuffle

    Need one of the next two
    n : int
        The size of the blocks that you're shuffling
    OR

    block_indices : list
        A list of length arr.shape[0] that specificies block identity for
        each timepoint


    Returns
    -------
    a_shuf : array
        The shuffled array.
    '''
    if block_indices is None:
        blks = np.ceil(float(arr.shape[0]) / n)
        blklst = np.arange(blks)
        block_indices = np.hstack([[i]*n for i in blklst])[:arr.shape[0]]
    else:
        blklst = np.unique(block_indices)
    np.random.shuffle(blklst)
    a_shuf = np.hstack([arr[block_indices==i] for i in blklst])
    return a_shuf
