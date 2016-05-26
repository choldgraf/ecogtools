"""Cross validation extras."""

import numpy as np
from sklearn import cross_validation as cval
from sklearn.cross_validation import PredefinedSplit


def permute_label_mapping(labs):
    """Permute an array of labels, leaving relative positioning intact.

    This will find the unique labels in `labs`, shuffle these, and build
    a mapping of old_label -> new_label.

    Parameters
    ----------
    labs : array-like
        The labels to be permuted. Unique labels will be randomly swapped
        with one another.

    Returns
    -------
    new_labs : array-like
        a version of labs where each number is randomly flipped with another
        number, though the overall structure of the labels is the same. E.g.,
        all 1s may now be 4s.
    """
    unique_labels = np.unique(labs)
    permuted_labels = np.random.permutation(unique_labels)
    random_mapping = {lab: newlab for lab, newlab
                      in zip(unique_labels, permuted_labels)}
    new_labs = np.array([random_mapping[ilab] for ilab in labs])
    return new_labs


def create_cv_from_trials(tnums, test_perc=.1, n_iter=5):
    """Create a cross validation object using trial numbers.

    This is similar to LeavePLabelOut, but it defines a stopping point and
    shuffles unique labels before doing the splits. This lets you keep
    datapoints together in train/test splits.

    Parameters
    ----------
    tnums : array, dtype int, shape(n_test)
        The labels to use for permutation.
    """
    tnums = tnums.squeeze()
    unique_labels = np.unique(tnums).squeeze()
    n_test = np.floor(unique_labels.shape[0] * test_perc)
    test_ixs = np.random.permutation(unique_labels)[:n_test * n_iter]
    test_ixs = test_ixs.reshape([n_iter, n_test])
    test_fold = np.zeros_like(tnums)
    for i, ifold in enumerate(test_ixs):
        for fld in ifold:
            test_fold[tnums == fld] = i
    cv = PredefinedSplit(test_fold)
    return cv
