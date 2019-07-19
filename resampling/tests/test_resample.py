import pytest
import numpy as np
from resampling.resample import Resample


@pytest.fixture
def data_2d():
    x, y = np.random.random((2, 10000))
    data = x + 2 * y
    return np.array([x, y]), data


def test_01_check_input_arrays():
    nfeatures = 3
    ndata = 100
    coordinates = tuple(x for x in np.random.random((nfeatures, ndata)))
    data = np.random.random(ndata)
    c, d, e, m = Resample._check_input_arrays(coordinates, data)
    assert c.shape == (nfeatures, ndata)
    assert d.size == ndata
    assert e is None
    assert m is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              error=data.copy())
    assert c.shape == (nfeatures, ndata)
    assert d.size == ndata
    assert np.allclose(d, e)
    assert m is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              mask=data > 0.5)
    assert c.shape == (nfeatures, ndata)
    assert d.size == ndata
    assert np.allclose(d > 0.5, m)
    assert e is None

    c, d, e, m = Resample._check_input_arrays(coordinates, data,
                                              mask=data > 0.5,
                                              error=data.copy())
    assert c.shape == (nfeatures, ndata)
    assert d.size == ndata
    assert np.allclose(d > 0.5, m)
    assert np.allclose(d, e)

    c, d, e, m = Resample._check_input_arrays(
        coordinates, np.vstack([data, data]))
    assert d.shape == (2, ndata)

    with pytest.raises(ValueError):
        _ = Resample._check_input_arrays(coordinates, data[:-1])
        _ = Resample._check_input_arrays(coordinates,
                                         np.zeros((3, 3, ndata)))
        _ = Resample._check_input_arrays(coordinates, data, error=data[:-1])
        _ = Resample._check_input_arrays(coordinates, data,
                                         mask=data[:-1] > 0.5)


def test_02_check_order():
    nfeatures = 3
    ndata = 100
    order = Resample._check_order(2, nfeatures, ndata)
    assert isinstance(order, np.ndarray)
    assert order.shape == ()
    order = Resample._check_order([2, 1, 1], nfeatures, ndata)
    assert isinstance(order, np.ndarray)
    assert order.shape == (nfeatures,)
    with pytest.raises(ValueError):
        _ = Resample._check_order(2, nfeatures, 1)
        _ = Resample._check_order(np.zeros(3, 3), nfeatures, ndata)
        _ = Resample._check_order(np.full(nfeatures + 1, 1), nfeatures, ndata)


def test_03_process_input_data(data_2d):
    coordinates, data = data_2d
    r = Resample(coordinates, data)
    r._process_input_data(data, None, None, negthresh=None, robust=10)
    assert r.mask.all()
    assert np.allclose(r.error, 1)
    assert not r.multi_set
    assert r.n_sets == 1

    r._process_input_data(np.vstack([data, data]), None, None,
                          negthresh=None, robust=10)
    assert r.multi_set
    assert r.n_sets == 2

    r._process_input_data(data, None, None, negthresh=None, robust=0.01)
    assert not r.mask.all()
    r._process_input_data(data - 100, None, None, negthresh=None, robust=None)
    assert r.mask.all()
    with pytest.raises(ValueError):
        r._process_input_data(data - 100, None, None, negthresh=0)
    r._process_input_data(data - np.mean(data), None, None,
                          negthresh=0, robust=None)
    assert not r.mask.all()

    error = data.copy()
    r._process_input_data(data, error, None, robust=0.01)
    invalid = ~r.mask
    assert np.isnan(r.data[invalid]).all()
    assert np.isnan(r.error[invalid]).all()

    mask = data > 0.5
    r._process_input_data(data, error, mask)
    assert not r.mask.all()
    invalid = ~r.mask
    assert np.isnan(r.data[invalid]).all()
    assert np.isnan(r.error[invalid]).all()
