from resampling.resample_utils import *
import numpy as np
import pytest


def test_polyexp():  # also tests `taylor`
    with pytest.raises(ValueError):
        polynomial_exponents(np.zeros((2, 2)))

    assert np.allclose(polynomial_exponents(3), np.array([[0], [1], [2], [3]]))
    assert np.allclose(polynomial_exponents([1, 2]),
                       [[0, 0], [1, 0],
                        [0, 1], [1, 1],
                        [0, 2]])
    assert np.allclose(polynomial_exponents(3, ndim=2),
                       [[0, 0], [1, 0], [2, 0], [3, 0],
                        [0, 1], [1, 1], [2, 1],
                        [0, 2], [1, 2],
                        [0, 3]])


def test_robust_mask():
    random = np.random.RandomState(41)
    # testing mask propagates
    data = random.rand(64, 64)
    test_mask = data < 0.9
    assert np.allclose(robust_mask(data, 3, mask=test_mask),
                       test_mask)

    # Testing zero threshold
    dnan = data.copy()
    dnan[~test_mask] = np.nan
    assert np.allclose(robust_mask(data, 0, mask=test_mask),
                       test_mask)

    # testing algorithm
    point_anomaly = data.copy()
    idx = np.nonzero(test_mask)
    point_anomaly[idx[0][0], idx[1][0]] = 1e3
    assert (robust_mask(point_anomaly, 3, mask=test_mask).sum() ==
            (test_mask.sum() - 1))

    # test axis
    row_anomaly = data.copy()
    row_anomaly[32] += 1e3
    assert (robust_mask(row_anomaly, 3, axis=0).sum() <
            robust_mask(row_anomaly, 3, axis=1).sum())

    with pytest.raises(ValueError):
        robust_mask(data, test_mask[0])


def test_scale_coordinates():
    random = np.random.RandomState(41)
    dvec = random.rand(3, 32) + 1
    dscalar = dvec[:, 0]

    scale = np.arange(3) + 1.0
    offset = np.arange(3) + 10.0

    # scale_forward_scalar
    x = scale_coordinates(dscalar, scale, offset, reverse=False)
    assert np.allclose(x, (dscalar - offset) / scale)

    # scale_reverse_scalar
    x = scale_coordinates(dscalar, scale, offset, reverse=True)
    assert np.allclose(x, dscalar * scale + offset)

    # scale_forward_vector
    x = scale_coordinates(dvec, scale, offset, reverse=False)
    assert np.allclose(x,
                       (dvec - offset[:, None]) / scale[:, None])

    # scale_reverse_vector
    x = scale_coordinates(dvec, scale, offset, reverse=True)
    assert np.allclose(x, dvec * scale[:, None] + offset[:, None])


def test_power_product():
    random = np.random.RandomState(41)
    exponents = polynomial_exponents(2, ndim=2)
    datasets = random.rand(2, 100)  # 2-D, 100 points

    # test tensor_power_product
    x = power_product(datasets, exponents)
    assert x.shape == (6, 100)
    assert np.allclose(x[5, 0], datasets[1, 0] ** 2)

    # test matrix_power_product
    x = power_product(datasets[:, 0], exponents)
    assert x.shape == (6,)
    assert np.allclose(x[4], datasets[0, 0] * datasets[1, 0])


def test_cull_members():
    nfeatures = 3
    ndata = 100
    delta = (np.random.random(nfeatures) - 0.3)
    offsets = np.random.random((nfeatures, ndata + 100))
    multiplier = np.random.random(nfeatures)
    members = np.arange(ndata) + 100

    return_indices = False
    expected = (multiplier[:, None] * offsets + delta[:, None]) ** 2
    evalue = expected[:, 100:].sum(0)
    expected = evalue <= 1

    mout = cull_members(offsets, members, multiplier, delta, return_indices)
    assert mout.size == expected.sum()
    assert expected[mout - 100].all()

    mout = cull_members(offsets, members, multiplier, delta, True)
    assert mout.size == expected.sum()
    assert expected[mout].all()


def test_linear_system_solve():
    random = np.random.RandomState(41)
    coordinates = random.rand(2, 100)
    data = random.rand(100)
    weightset = random.rand(100)
    exponents = polynomial_exponents(2, ndim=2)
    powerset = power_product(coordinates, exponents)
    visitor_power = powerset.mean(axis=1)
    x = linear_system_solve(
        powerset, data, weightset, visitor_power)
    assert np.allclose(x, 0.4852216645369541)


def test_weighted_mean():
    data = np.arange(10, dtype=float)
    weights = (1 / (np.arange(10) + 1)) ** 2
    mean, var = weighted_mean(data, weights)
    assert np.allclose(mean, 0.8899401472010013)
    assert np.allclose(var, 0.6452579827864142)

    mean, var = weighted_mean(data, weights * 0)
    assert np.isnan(mean)
    assert var == np.inf


def test_calculate_set_weights():
    set_error = np.asarray([2.0])
    set_dweight = np.asarray([3.0])

    wt, wt2 = calculate_set_weights(set_error, set_dweight, True)
    assert np.allclose(wt, 1/6)
    assert np.allclose(wt2, 1/12)


def test_mask_count():
    mask = np.random.random(100) > 0.5
    assert mask_count(mask) == mask.sum()


def test_get_fit_range():
    data = np.random.random(100)
    mask = data > 0.5

    nmask = mask.sum()
    assert get_fit_range(0, data, mask) == (np.inf, nmask)
    assert get_fit_range(2.0, data, np.full_like(mask, False)) == (
        np.inf, 0)

    threshold, count = get_fit_range(2.0, data, mask)
    assert count == nmask
    dmask = data[mask]
    dmean = dmask.mean()
    rms = np.sqrt(np.sum((dmask - dmean) ** 2) / (count - 1))
    assert np.allclose(threshold, rms * 2.0)


def test_get_fit_range_with_error():
    data = np.random.random(100)
    mask = data > 0.5
    error = np.random.random(100)

    nmask = mask.sum()
    assert get_fit_range_with_error(
        0, data, mask, error) == (np.inf, nmask)
    assert get_fit_range_with_error(
        2.0, data, np.full_like(mask, False), error) == (np.inf, 0)

    assert get_fit_range_with_error(
        2.0, data, mask, np.zeros_like(data)) == get_fit_range(
        2.0, data, mask)

    threshold, _ = get_fit_range_with_error(
        1.0, data, mask, np.full_like(data, 1e6))
    assert np.allclose(threshold, 1e6)


def test_offsets_from_center():
    coordinates = np.random.random((3, 100))
    center = np.mean(coordinates, axis=1)
    assert np.allclose(offsets_from_center(coordinates, center),
                       coordinates - center[:, None])


def test_calculate_distance_weights():

    coordinates = np.random.random((3, 100))
    center = np.mean(coordinates, axis=1)

    # no distance weighting
    assert np.allclose(calculate_distance_weights(
        coordinates, center, np.full(1, 0.0)), 1)

    expected = np.exp(
        (((coordinates - center[:, None]) ** 2) / 2).sum(0))
    assert np.allclose(
        expected, calculate_distance_weights(
            coordinates, center, np.asarray([2.0])))


def test_check_edge_by_com():
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    # Center of mass is zero here
    center = np.asarray([4.5, 4.5])
    threshold = np.asarray([0.5, 0.5])
    assert check_edge_by_com(coordinates, center, threshold)
    # At the edge threshold (0.5) returns True
    center = np.asarray([5.0, 5.0])
    assert check_edge_by_com(coordinates, center, threshold)

    # slightly over edge threshold
    center += 0.1
    assert not check_edge_by_com(coordinates, center, threshold)

    # If even one dimension is over the threshold it should fail
    center = np.asarray([4.5, 5.1])
    assert not check_edge_by_com(coordinates, center, threshold)

    # Check threshold = 0 or 1 always passes
    assert check_edge_by_com(coordinates, center + 1000, threshold * 0)
    assert check_edge_by_com(coordinates, center + 1000, threshold * 0 + 1)


def test_check_edge_by_com_distance():
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    # Center of mass is zero here
    center = np.asarray([4.5, 4.5])
    threshold = np.asarray([0.5, 0.5])
    assert check_edge_by_com_distance(coordinates, center, threshold)
    # At the edge threshold (0.5) returns False (COM ** 2 = 2)
    center = np.asarray([5.0, 5.0])
    assert not check_edge_by_com_distance(coordinates, center, threshold)

    # The overall distance is 0.5 here (0.5 ** 2 = 0.25)
    center = np.asarray([4.75, 4.75])
    assert check_edge_by_com_distance(coordinates, center, threshold)

    # This should fail too since distance = 0.6
    center = np.asarray([4.5, 5.1])
    assert not check_edge_by_com_distance(coordinates, center, threshold)

    # Check threshold = 0 or 1 always passes
    assert check_edge_by_com(coordinates, center + 1000, threshold * 0)
    assert check_edge_by_com(coordinates, center + 1000, threshold * 0 + 1)


def test_edge_by_range():
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    # Center of mass is zero here
    center = np.asarray([4.5, 4.5])
    threshold = np.asarray([0.5, 0.5])
    assert check_edge_by_range(coordinates, center, threshold)

    # check_edge_by_range should only ever fail near the border
    center = np.asarray([8.5, 8.5])
    assert check_edge_by_range(coordinates, center, threshold)
    center = np.asarray([8.6, 8.6])
    assert not check_edge_by_range(coordinates, center, threshold)

    # Check threshold = 0 or 1 always passes
    assert check_edge_by_range(coordinates, center + 1000, threshold * 0)
    assert check_edge_by_range(coordinates, center + 1000, threshold * 0 + 1)


def test_check_edges():
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    center = np.asarray([4.5, 4.5])
    threshold = np.asarray([0.5, 0.5])
    for algorithm in range(1, 5):
        assert check_edges(algorithm, coordinates, center, threshold)


def test_edges_maxorder():  # also checks `sides_max_order`
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    mask = np.full(coordinates.shape[1], True)
    order = np.array([2, 3])

    # Check corner cases (literally)
    center = np.array([1.0, 2.0])
    assert np.allclose(
        edges_maxorder(coordinates, center, mask, order, False), [1, 2])
    assert edges_maxorder(coordinates, center, mask, order, True)[0] == -1

    center += 0.5
    assert np.allclose(
        edges_maxorder(coordinates, center, mask, order, False), [2, 3])

    center = np.array([8.0, 7.0])
    assert np.allclose(
        edges_maxorder(coordinates, center, mask, order, False), [1, 2])
    assert edges_maxorder(coordinates, center, mask, order, True)[0] == -1

    center -= 0.5
    assert np.allclose(
        edges_maxorder(coordinates, center, mask, order, False), [2, 3])

    # Check mask
    mask[coordinates[0] > 8] = False
    assert edges_maxorder(coordinates, center, mask, order, True)[0] == -1


def test_counts_maxorder():
    mask = np.full(100, True)
    minimum_points = 101
    order = np.full(1, 2)
    nfeatures = 3
    assert (counts_maxorder(mask, order, minimum_points, nfeatures, False)[0]
            == 3)
    assert (counts_maxorder(mask, order, minimum_points, nfeatures, True)[0]
            == -1)

    mask[:] = False
    assert (counts_maxorder(mask, order, minimum_points, nfeatures, False)[0]
            == -1)

    mask[:] = True
    order = np.full(3, 2)
    assert (counts_maxorder(mask, order, minimum_points, nfeatures, False)[0]
            == -1)
    minimum_points = 100
    assert np.allclose(
        counts_maxorder(mask, order, minimum_points, nfeatures, False),
        [2, 2, 2])


def test_extrapolate_maxorder():  # also checks `unique_values_max_order`
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    mask = np.full(coordinates.shape[1], True)
    order = np.array([2, 3])

    assert np.allclose(
        extrapolate_maxorder(coordinates, mask, order, False), [2, 3])
    assert np.allclose(
        extrapolate_maxorder(coordinates, mask, order, True), [2, 3])

    mask[coordinates[1] > 2] = False
    assert np.allclose(
        extrapolate_maxorder(coordinates, mask, order, False), [2, 2])
    assert extrapolate_maxorder(coordinates, mask, order, True)[0] == -1


def test_check_orders():
    y, x = np.mgrid[:10, :10]
    y = y.astype(float)
    x = x.astype(float)
    coordinates = np.vstack([x.ravel(), y.ravel()])
    order = np.full(1, 2)
    minimum_points = 5
    mask = np.full(coordinates.shape[1], True)
    center = np.asarray([4.5, 4.5])
    required = False
    for algorithm in range(4):
        assert check_orders(algorithm, coordinates, center, mask, order,
                            minimum_points, required)


def test_prune_equation_arrays():
    mask = np.random.random(100) > 0.2
    counts = mask.sum()
    data = np.random.random(100)
    powers = np.random.random((3, 100))
    error = np.random.random(100)
    weights = np.random.random(100)

    d, p, e, w = prune_equation_arrays(
        counts, mask, data, powers, error, weights)
    assert d.size == counts
    assert p.shape == (3, counts)
    assert e.size == counts
    assert w.size == counts

    # single valued error
    error = error[:1]
    d, p, e, w = prune_equation_arrays(
        counts, mask, data, powers, error, weights)
    assert d.size == counts
    assert p.shape == (3, counts)
    assert e.size == counts
    assert w.size == counts
    assert np.allclose(e, error[0])


def test_no_set_solution():
    nsets = 3
    ndata = 100
    shape = nsets, ndata
    fit_out = np.full(shape, np.nan)
    error_out = np.full(shape, np.nan)
    counts_out = np.ones(shape, dtype=int)
    cval = -1.0
    get_error = True
    get_counts = True
    visitor_id = 2
    dataset = 1
    no_set_solution(fit_out, error_out, counts_out, visitor_id, dataset,
                    cval, get_error, get_counts)
    assert fit_out[dataset, visitor_id] == -1
    assert error_out[dataset, visitor_id] == 0
    assert counts_out[dataset, visitor_id] == 0
