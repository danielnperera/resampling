import bottleneck
import math
import numpy as np
import numba as nb
from numba import njit
import warnings

_condition_limit = 1 / np.finfo(np.float).eps
_fast_flags = {'nsz', 'nnan', 'ninf'}


def polynomial_exponents(order, ndim=None):
    """
    Returns exponents for given polynomial orders in arbitrary dimensions

    Parameters
    ----------
    order : int or array_like of int
        Polynomial order for which to generate exponents.  If an array
        will create full polynomial exponents over all len(order)
        dimensions.

    ndim : int, optional
        If set, return Taylor expansion for `ndim` dimensions for
        the given `order` if `order` is not an array.

    Returns
    -------
    exponents : numpy.ndarray
        (n_coefficients, n_dimensions) array of polynomial exponents.

    Examples
    --------
    >>> polynomial_exponents(3)
    array([[0],
           [1],
           [2],
           [3]])

    >>> polynomial_exponents([1, 2])
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1],
           [0, 2]])

    >>> polynomial_exponents(3, ndim=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [3, 0],
           [0, 1],
           [1, 1],
           [2, 1],
           [0, 2],
           [1, 2],
           [0, 3]])
    """
    order = np.atleast_1d(np.asarray(order, dtype=int))
    if order.ndim > 1:
        raise ValueError("order must have 0 or 1 dimensions")
    if order.size > 1:
        ndim = order.size
        limit_orders = True
        max_order = order.max()
    else:
        if ndim is None:
            ndim = 1
        limit_orders = False
        max_order = order[0]

    exponents = np.asarray([list(e) for e in taylor(max_order, int(ndim))])
    exponents = np.flip(exponents, axis=-1)

    if limit_orders and ndim > 1:
        keep = np.logical_not(np.any(exponents > order[None], axis=1))
        exponents = exponents[keep]

    return exponents


def taylor(order, n):
    """
    Taylor expansion generator for Polynomial exponents

    Parameters
    ----------
    order : int
        Order of Polynomial
    n : int
        Number of variables to solve for

    Yields
    ------
    n-tuple of int
        The next polynomial exponent
    """
    if n == 0:
        yield()
        return
    for i in range(order + 1):
        for result in taylor(order - i, n - 1):
            yield (i,) + result


def robust_mask(data, threshold, mask=None, axis=None):
    """
    Computes a mask based a thresholding parameter and data MAD.

    Procedure
    ---------
    Calculates a robust mask based on the input data and optional input mask.
    If `threshold` is greater than zero, the dataset is searched for outliers.
    A data point is identified as an outlier if |x_i - x_med|/MAD > threshold,
    where x_med is the median, MAD is the median absolute deviation defined as
    1.482 * median(|x_i - x_med|).

    Parameters
    ----------
    data : array_like of float
        (shape1) Data on which to calculate moments
    mask : array_like of bool
        (shape1) Mask to apply to data
    threshold : float, optional
        Sigma threshold over which values are identified as outliers
    axis : int, optional
        Axis over which to calculate statistics

    Returns
    -------
    dict or numpy.ndarray
        If `get_mask` is False, returns a dictionary containing the
        following statistics: mean, var, stddev, skew, kurt, stderr,
        mask.  Otherwise, returns the output mask.
    """
    d = np.asarray(data, dtype=float).copy()
    valid = np.isfinite(d)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != d.shape:
            raise ValueError("data and mask shape mismatch")
        valid &= mask

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        if threshold is not None and threshold > 0:
            d[~valid] = np.nan
            if axis is None:
                med = bottleneck.nanmedian(d)
                mad = 1.482 * bottleneck.nanmedian(np.abs(d - med))
            else:
                med = np.expand_dims(bottleneck.nanmedian(d, axis=axis), axis)
                mad = np.expand_dims(
                    1.482 * bottleneck.nanmedian(
                        np.abs(d - med), axis=axis), axis)

            ratio = np.abs(d - med) / mad
            valid &= ratio <= threshold

    return valid


def scale_coordinates(coordinates, scale, offset, reverse=False):
    scalar = coordinates.ndim == 1
    if reverse:
        if scalar:
            return scale_reverse_scalar(coordinates, scale, offset)
        else:
            return scale_reverse_vector(coordinates, scale, offset)
    else:
        if scalar:
            return scale_forward_scalar(coordinates, scale, offset)
        else:
            return scale_forward_vector(coordinates, scale, offset)


def power_product(values, exponents):
    if values.ndim == 2:
        return tensor_power_product(values, exponents)
    else:
        return matrix_power_product(values, exponents)


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def scale_forward_vector(coordinates, scale, offset):
    features, ndata = coordinates.shape
    result = np.empty((features, ndata))
    for k in range(features):
        for i in range(ndata):
            result[k, i] = (coordinates[k, i] - offset[k]) / scale[k]
    return result


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def scale_forward_scalar(coordinates, scale, offset):
    features = coordinates.size
    result = np.empty(features)
    for k in range(features):
        result[k] = (coordinates[k] - offset[k]) / scale[k]
    return result


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def scale_reverse_vector(coordinates, scale, offset):

    features, ndata = coordinates.shape
    result = np.empty((features, ndata))
    for k in range(features):
        for i in range(ndata):
            result[k, i] = coordinates[k, i] * scale[k] + offset[k]
    return result


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def scale_reverse_scalar(coordinates, scale, offset):
    features = coordinates.size
    result = np.empty(features)
    for k in range(features):
        result[k] = coordinates[k] * scale[k] + offset[k]
    return result


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def matrix_power_product(values, exponents):
    n_coeffs, n_dimensions = exponents.shape
    pp = np.empty(n_coeffs)
    for i in range(n_coeffs):
        x = 1.0
        for j in range(n_dimensions):
            val = values[j]
            exponent = exponents[i, j]
            val_e = 1.0
            for l in range(exponent):
                val_e *= val
            x *= val_e
        pp[i] = x
    return pp


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def tensor_power_product(values, exponents):
    n_coeffs, n_dimensions = exponents.shape
    n_data = values.shape[1]
    pp = np.empty((n_coeffs, n_data))

    for k in range(n_data):
        for i in range(n_coeffs):
            x = 1.0
            for j in range(n_dimensions):
                val = values[j, k]
                exponent = exponents[i, j]
                val_e = 1.0
                for l in range(exponent):
                    val_e *= val
                x *= val_e
            pp[i, k] = x
    return pp


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def cull_members(offsets, members, multiplier, delta, return_indices):

    features = delta.size
    for k in range(features):
        if multiplier[k] != 0:
            break
        elif delta[k] != 0:
            break
    else:
        if return_indices:
            result = np.empty(members.size, dtype=nb.i8)
            for i in range(members.size):
                result[i] = i
            return result
        else:
            return members

    nmembers = members.size
    keep = np.empty(nmembers, dtype=nb.i8)
    nfound = 0
    for i in range(nmembers):
        d = 0.0
        member = members[i]
        offset = offsets[:features, member]
        for k in range(features):
            doff = multiplier[k] * offset[k] + delta[k]
            doff *= doff
            d += doff
            if d > 1:
                break
        else:
            if return_indices:
                keep[nfound] = i
            else:
                keep[nfound] = member

            nfound += 1

    return keep[:nfound]


@njit(nogil=True, cache=True)
def linear_system_solve(powerset, dataset, weightset, visitor_power):

    ncoeffs, ndata = powerset.shape
    alpha = np.empty((ncoeffs, ndata))
    beta = np.empty(ncoeffs)
    amat = np.empty((ncoeffs, ncoeffs))

    for i in range(ncoeffs):
        b = 0.0
        for k in range(ndata):
            w = weightset[k]
            wa = w * powerset[i, k]
            b += wa * w * dataset[k]
            alpha[i, k] = wa
        beta[i] = b

    for i in range(ncoeffs):
        for j in range(i, ncoeffs):
            asum = 0.0
            for k in range(ndata):
                asum += alpha[i, k] * alpha[j, k]
            amat[i, j] = asum
            if i != j:
                amat[j, i] = asum

    coefficients = np.linalg.solve(amat, beta)
    result = 0.0
    for i in range(ncoeffs):
        result += coefficients[i] * visitor_power[i]
    return result


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def weighted_mean(data, inverse_squared_weights):
    n = data.size
    wsum = 0.0
    for i in range(n):
        wsum += inverse_squared_weights[i]
    if wsum == 0:
        return np.nan, np.nan

    dsum = 0.0
    for i in range(n):
        dsum += data[i] * inverse_squared_weights[i]

    return dsum / wsum, 1.0 / wsum


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def calculate_set_weights(set_error, set_dweight, calculate_variance):
    # inverts it too
    n = set_error.size
    weights = np.empty(n)
    if calculate_variance:
        weights2 = np.empty(n)
    else:
        weights2 = weights

    for i in range(n):
        e = 1.0 / set_error[i]
        weights[i] = e / set_dweight[i]
        if calculate_variance:
            weights2[i] = weights[i] * e
    return weights, weights2


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def mask_count(mask):
    count = 0
    for i in range(mask.size):
        count += mask[i]
    return count


@njit(nogil=True, cache=True)
def get_fit_range(threshold, data, mask):

    n = data.size
    count = 0
    for i in range(n):
        count += mask[i]

    if count < 2 or not threshold:
        return np.inf, count

    dmean = 0.0
    for i in range(n):
        if mask[i]:
            dmean += data[i]
    dmean /= count

    dsum = 0.0
    for i in range(n):
        if mask[i]:
            d = data[i] - dmean
            d *= d
            dsum += d
    dsum /= count - 1
    return math.sqrt(dsum) * threshold, count


@njit(nogil=True, cache=True)
def get_fit_range_with_error(threshold, data, mask, error):

    n = data.size
    count = 0
    for i in range(n):
        count += mask[i]

    if count < 2 or not threshold:
        return np.inf, count

    dmean = 0.0
    emean = 0.0
    for i in range(n):
        if mask[i]:
            dmean += data[i]
            emean += error[i]
    dmean /= count
    emean /= count

    dsum = 0.0
    for i in range(n):
        if mask[i]:
            d = data[i] - dmean
            d *= d
            dsum += d
    dsum /= count - 1
    dsig = math.sqrt(dsum)
    if dsig < emean:
        dsig = emean

    return dsig * threshold, count


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def offsets_from_center(coordinates, center):
    features, npoints = coordinates.shape
    offsets = np.empty((features, npoints))
    for k in range(features):
        for i in range(npoints):
            offsets[k, i] = coordinates[k, i] - center[k]
    return offsets


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def calculate_distance_weights(coordinates, center, alpha):
    features, n = coordinates.shape
    weights = np.empty(n)
    symmetric = alpha.size == 1

    if symmetric and alpha[0] == 0:
        for i in range(n):
            weights[i] = 1.0
        return weights
    else:
        for i in range(n):
            weights[i] = 0.0

    for k in range(features):
        a = alpha[0] if symmetric else alpha[k]
        if a == 0:
            continue
        for i in range(n):
            d = coordinates[k, i] - center[k]
            d *= d
            d /= a
            weights[i] += d

    for i in range(n):
        weights[i] = math.exp(weights[i])

    return weights


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def check_edges(algorithm, coordinates, center, threshold):
    if algorithm == 1:
        return check_edge_by_com_distance(coordinates, center, threshold)
    elif algorithm == 2:
        return check_edge_by_com(coordinates, center, threshold)
    elif algorithm == 3:
        return check_edge_by_range(coordinates, center, threshold)
    else:
        return True


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def check_edge_by_com(coordinates, center, threshold):
    features, ndata = coordinates.shape
    max_deviation = 1.0 - threshold
    for k in range(features):
        if threshold[k] == 0 or threshold[k] == 1:
            continue
        com = 0.0
        for i in range(ndata):
            com += coordinates[k, i] - center[k]
        com /= ndata
        if abs(com) > max_deviation[k]:
            return False
    else:
        return True


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def check_edge_by_com_distance(coordinates, center, threshold):
    features, ndata = coordinates.shape
    max_deviation = 1.0 - threshold
    offset = 0.0
    n2 = ndata * ndata
    for k in range(features):
        if threshold[k] == 0 or threshold[k] == 1:
            continue
        com = 0.0
        for i in range(ndata):
            com += coordinates[k, i] - center[k]

        com /= max_deviation[k]
        offset += com * com

    offset /= n2
    if offset > 1:
        return False
    else:
        return True


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def check_edge_by_range(coordinates, center, threshold):
    features, ndata = coordinates.shape
    negthresh = threshold * -1
    for k in range(features):
        left_found = False
        right_found = False
        if threshold[k] == 0 or threshold[k] == 1:
            continue
        for i in range(ndata):
            offset = coordinates[k, i] - center[k]
            if offset < 0:
                if left_found:
                    continue
                else:
                    left_found = offset <= negthresh[k]
            elif offset > 0:
                if right_found:
                    continue
                else:
                    right_found = offset >= threshold[k]
            if left_found and right_found:
                break
        else:
            return False
    else:
        return True


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def check_orders(algorithm, coordinates, center, mask, order,
                 minimum_points, required):
    if algorithm == 1:  # check enough points either side
        return edges_maxorder(coordinates, center, mask, order, required)
    elif algorithm == 2:  # check enough points overall
        return counts_maxorder(mask, order, minimum_points,
                               coordinates.shape[0], required)
    elif algorithm == 3:  # allow extrapolation if visitor outside local span
        return extrapolate_maxorder(coordinates, mask, order, required)
    else:
        return np.full(1, -1, dtype=nb.i8)  # must be checked


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def edges_maxorder(coordinates, center, mask, order, required):
    """Checks the maximum allowable order based on 2-sided feature analysis

    Checks the number of unique `coordinates` to the left and right of
    `center` over all features.
    """

    features = coordinates.shape[0]
    norders = order.size
    symmetric = norders == 1
    order_out = np.empty(order.size, dtype=nb.i8)
    for i in range(norders):
        order_out[i] = order[i]

    for k in range(features):
        idx = 0 if symmetric else k
        o = order_out[idx]
        minside = sides_max_order(
            o, center[k], coordinates[k], mask, required)
        if minside < 0:
            order_out[0] = -1
            break
        order_out[idx] = minside

    return order_out


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def sides_max_order(order, center, coordinates, mask, required):
    """Support function for edges_maxorder"""
    if order == 0:
        return 0
    left = 0
    right = 0
    left_found = False
    right_found = False
    unique_left = np.empty(order)
    unique_right = np.empty(order)

    for i in range(coordinates.size):
        if not mask[i]:
            continue
        offset = coordinates[i] - center
        if offset < 0:
            if left_found:
                continue
            elif left == 0:
                unique_left[0] = offset
                left = 1
            else:
                for j in range(left):
                    if unique_left[j] == offset:
                        break
                else:
                    unique_left[left] = offset
                    left += 1
            if left >= order:
                left_found = True

        elif offset > 0:
            if right_found:
                continue
            elif right == 0:
                unique_right[0] = offset
                right = 1
            else:
                for j in range(right):
                    if unique_right[j] == offset:
                        break
                else:
                    unique_right[right] = offset
                    right += 1
            if right >= order:
                right_found = True

        if left_found and right_found:
            return order
    else:
        if required:
            return -1
        elif left < right:
            return left
        else:
            return right


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def counts_maxorder(mask, order, minimum_points, nfeatures, required):
    """"""

    norders = order.size
    symmetric = norders == 1
    order_out = np.empty(order.size, dtype=nb.i8)
    for i in range(norders):
        order_out[i] = order[i]

    msum = 0
    for k in range(mask.size):
        msum += mask[k]
        if msum == minimum_points:
            break
    else:
        if required or not symmetric:
            order_out[0] = -1
        else:
            limit = msum ** (1.0 / nfeatures) - 1
            if limit < 0:
                order_out[0] = -1
            else:
                order_out[0] = nb.i8(limit)

    return order_out


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def extrapolate_maxorder(coordinates, mask, order, required):

    features = coordinates.shape[0]
    norders = order.size
    symmetric = norders == 1
    order_out = np.empty(order.size, dtype=nb.i8)
    for i in range(norders):
        order_out[i] = order[i]

    for k in range(features):
        idx = 0 if symmetric else k
        o = order_out[idx]
        minside = unique_values_max_order(
            o, coordinates[k], mask, required)
        if minside < 0:
            order_out[0] = -1
            break
        order_out[idx] = minside

    return order_out


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def unique_values_max_order(order, coordinates, mask, required):
    if order == 0:
        return 0
    max_order = -1
    unique_values = np.empty(order + 1)

    for i in range(coordinates.size):
        if not mask[i]:
            continue
        value = coordinates[i]
        for j in range(max_order + 1):
            if unique_values[j] == value:
                break
        else:
            max_order += 1
            unique_values[max_order] = value
        if max_order >= order:
            return max_order
    else:
        if required:
            return -1
        else:
            return max_order


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def prune_equation_arrays(counts, equation_mask,
                          equation_data, powers,
                          equation_error,
                          equation_dweights):

    ncoeffs, n = powers.shape
    powerset = np.empty((ncoeffs, counts))
    dataset = np.empty(counts)
    errorset = np.empty(counts)
    dweightset = np.empty(counts)

    single_error = equation_error.size == 1
    single_dweight = equation_dweights.size == 1

    found = 0
    for i in range(n):
        if equation_mask[i]:
            dataset[found] = equation_data[i]
            for j in range(ncoeffs):
                powerset[j, found] = powers[j, i]
            if single_error:
                errorset[found] = equation_error[0]
            else:
                errorset[found] = equation_error[i]
            if single_dweight:
                dweightset[found] = equation_dweights[0]
            else:
                dweightset[found] = equation_dweights[i]
            found += 1
    return dataset, powerset, errorset, dweightset


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def no_set_solution(fit_out, error_out, counts_out, visitor_id, dataset,
                    cval, get_error, get_counts):
    fit_out[dataset, visitor_id] = cval
    if get_error:
        error_out[dataset, visitor_id] = np.nan
    if get_counts:
        counts_out[dataset, visitor_id] = 0


@njit(nogil=True, cache=True, fastmath=_fast_flags)
def solve_visitor(data, mask, error, cval, get_error, get_counts,
                  fit_out, error_out, counts_out,
                  visitor_id, visitor, visitor_coordinates, visitor_powers,
                  local_submembers, local_coordinates, local_powers,
                  order, order_idx, order_algorithm_idx, order_varies,
                  required, minimum_points, check_order0,
                  edge_algorithm_idx, edge_threshold, alpha,
                  fit_threshold, mean_only, get_mean):

    n_sets = data.shape[0]
    if local_submembers.size == 0:
        for dataset in range(n_sets):
            no_set_solution(fit_out, error_out, counts_out, visitor_id,
                            dataset, cval, get_error, get_counts)
        return

    local_subcoord = local_coordinates[:, local_submembers]
    visitor_subcoord = visitor_coordinates[:, visitor_id]

    # Do we need to remove edge effects?
    if not check_edges(edge_algorithm_idx, local_subcoord,
                       visitor_subcoord, edge_threshold):
        for dataset in range(n_sets):
            no_set_solution(fit_out, error_out, counts_out, visitor_id,
                            dataset, cval, get_error, get_counts)
        return

    # Calculate distance weights if needed (not needed: alpha = [0])
    dweights = calculate_distance_weights(
        local_subcoord, visitor_subcoord, alpha)

    # Calculate orders
    subdata = data[:, local_submembers]
    submask = mask[:, local_submembers]
    error_weighting = error.shape[1] > 1
    if error_weighting:
        suberror = error[:, local_submembers]
    else:
        suberror = error

    sublocal_powers = local_powers[:, local_submembers]
    subvisitor_powers = visitor_powers[:, visitor]
    max_deviation = np.inf
    check_fit = fit_threshold > 0

    for dataset in range(data.shape[0]):

        set_mask = submask[dataset]
        set_order = check_orders(
            order_algorithm_idx, local_subcoord, visitor_subcoord,
            set_mask, order,  minimum_points, required)

        if set_order[0] == -1:
            no_set_solution(fit_out, error_out, counts_out, visitor_id,
                            dataset, cval, get_error, get_counts)
            continue

        if check_order0:
            mean_only = False
            for o in set_order:
                if o != 0:
                    break
            else:
                mean_only = True

        set_data = subdata[dataset]
        set_error = suberror[dataset]
        if check_fit:
            if error_weighting:
                max_deviation, counts = get_fit_range_with_error(
                    fit_threshold, set_data, set_mask, set_error)
            else:
                max_deviation, counts = get_fit_range(
                    fit_threshold, set_data, set_mask)
        else:
            counts = mask_count(set_mask)

        if counts == 0:
            no_set_solution(fit_out, error_out, counts_out, visitor_id,
                            dataset, cval, get_error, get_counts)
            continue

        if get_counts:
            counts_out[dataset, visitor_id] = counts

        if order_varies:
            oidx = order_idx[set_order[0]: set_order[0] + 2]
            set_visitor_power = subvisitor_powers[oidx[0]: oidx[1]]
            set_local_power = sublocal_powers[oidx[0]: oidx[1]]
        else:
            set_visitor_power = subvisitor_powers
            set_local_power = sublocal_powers

        set_data, set_local_power, set_error, set_dweights = \
            prune_equation_arrays(counts, set_mask, set_data, set_local_power,
                                  set_error, dweights)

        set_weight, set_weight2 = calculate_set_weights(
            set_error, set_dweights, get_mean)

        if get_mean:
            set_mean, set_variance = weighted_mean(set_data, set_weight2)
            if get_error:
                error_out[dataset, visitor_id] = math.sqrt(set_variance)
        else:
            set_mean = np.nan

        if mean_only:
            fit_out[dataset, visitor_id] = set_mean
            continue

        fitted = linear_system_solve(
            set_local_power, set_data, set_weight, set_visitor_power)
        if check_fit:
            deviation = abs(fitted - set_mean)
            if deviation < max_deviation:
                fit_out[dataset, visitor_id] = fitted
            else:
                fit_out[dataset, visitor_id] = set_mean
        else:
            fit_out[dataset, visitor_id] = fitted
