import numpy as np
from .. import car, delay_timeseries
from numpy.testing import assert_almost_equal, assert_equal


def test_car():
    """Test common average reference."""
    groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    data = np.tile(groups, (25, 1))

    out = car(data, grouping=groups)
    assert_equal((out != 0).sum(),  0)

    # Make sure electrode exclusion is correct
    exclude_elecs = np.zeros_like(groups).astype(bool)
    exclude_elecs[[0, 4]] = 1
    for func in [np.mean, np.median]:
        out, com = car(data, grouping=groups, exclude_elecs=exclude_elecs,
                       return_averages=True, agg_func=func)
        for ix in [0, 4]:
            assert_almost_equal(out[:, ix]+com[:, groups[ix]-1],
                                data[:, ix])


def test_delay_timeseries():
    """Test for the custom time series lag function."""
    times = np.arange(1000).reshape([1, -1])
    sfreq = 1000
    delays = [0, .1, .2, .3]
    delayed_times = delay_timeseries(times, sfreq, delays)
    assert_equal(times[0], delayed_times[0])
    for i, idel in enumerate(delays[1:]):
        assert_equal(times[0, :-sfreq*idel], delayed_times[i+1, sfreq*idel:])
