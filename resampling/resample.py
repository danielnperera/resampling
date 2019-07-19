import bottleneck as bn
from numba import prange
import numpy as np
import warnings
from .resample_utils import scale_coordinates, solve_visitor, robust_mask
from .grid import ResampleGrid
from .tree import Rtree


class Resample(object):

    def __init__(self, coordinates, data, error=None, mask=None,
                 window=None, order=1, fix_order=True, robust=None,
                 negthresh=None, mode='edges'):
        """
        Resample data using local polynomial fits

        Parameters
        ----------
        coordinates : array_like of float
            (nfeatures, ndata) array of independent values.  A local
            internal copy will be created if it is not a numpy.float64
            type.
        data : array_like of float
            (nsets, ndata) or (ndata,) array of dependent values.
            If multiple (nsets) sets of data are supplied, then nequation
            solutions will be calculated at each resampling point.
        error : array_like of float, optional
            (nequations, ndata) or (ndata,) array of error (1-sigma) values
            associated with the `data` array.  `error` will be used to
            weight fits and also be propagated to the output error values.
            If not supplied, each data point is assumed to have an associated
            error value of 1.0.
        mask : array_like of bool, optional
            (nequations, data) or (ndata,) array of bool where True indicates
            a valid data point that can be included the fitting and False
            indicates data points that should be excluded from the fit.
            Masked points will be reflected in the output counts array.
        window : array_like or float or int, optional
            (nfeatures,) array or single float value specifying the maximum
            euclidean distance of a data sample from a resampling point such
            that it can be included in a local fit.  `window` may be declared
            for each feature.  For example, when fitting 2-dimensional (x, y)
            data, a window of 1.0 would create a circular fitting window
            around each resampling point, whereas a window of (1.0, 0.5)
            would create an eliptical fitting window with a semi-major axis
            of 1.0 in x and semi-minor axis of 0.5 in y.  If not supplied,
            `window` is calculated based on an estimate of the median
            population density of the data for each feature.
        order : array_like or int, optional
            (nfeatures,) array of single integer value specifying the
            polynomial fit order for each feature.
        fix_order : bool, optional
            In order for local polynomial fitting to occur, the default
            requirement is that nsamples >= (order + 1) ** nfeatures,
            where nsamples is the number of data samples within `window`.
            If `fix_order` is True and this condition is not met, then
            local fitting will be aborted for that point and a value of
            `cval` will be returned instead.  If `fix_order` is False,
            then `order` will be reduced to the maximum value where this
            condition can be met.  NOTE: this is only available if
            `order` is symmetrical. i.e. it was passed in as a single
            integer to be applied across all features.  Otherwise, it is
            unclear as to which feature order should be reduced to meet
            the condition.
        robust : float, optional
            Specifies an outlier rejection threshold for `data`.
            A data point is identified as an outlier if
            |x_i - x_med|/MAD > `robust`, where x_med is the median,
            and MAD is the Median Absolute Deviation defined as
            1.482 * median(|x_i - x_med|).
        negthresh : float, optional
            Specifies a negative value rejection threshold such that
            data < (-stddev(data) * negthresh) will be excluded from
            the fit.
        mode : str, optional
            The type of check to perform on whether the sample distribution
            for each resampling point is adequate to derive a polynomial fit.
            Depending on `order` and `fix_order`, if the distribution does
            not meet the criteria for `mode`, either the fit will be aborted,
            returning a value of `cval` or the fit order will be reduced.
            Available modes are:
                'edges': Require that there are `order` samples in both
                    the negative and positive directions of each feature
                    from the resampling point.
                'counts': Require that there are (order + 1) ** nfeatures
                    samples within the `window` of each resampling point.
                'extrapolate': Attempt to fit regardless of the sample
                    distribution.
            Note that 'edges' is the most robust mode as it ensures
            that no singular values will be encountered during the
            least-squares fitting of polynomial coefficients.

        Raises
        ------
        ValueError : Invalid inputs to __init__ or __call__
        """
        coordinates, data, error, mask = self._check_input_arrays(
            coordinates, data, error=error, mask=mask)
        self._nfeatures, self._nsamples = coordinates.shape
        order = self._check_order(order, self._nfeatures, self._nsamples)

        self._ndatasets = 0
        self._multiset = None
        self._valid_sets = None
        self.data = None
        self.error = None
        self._error_weighted = None
        self.mask = None
        self._process_input_data(
            data, error, mask, negthresh=negthresh, robust=robust)

        scaled_coordinates = self._scale_to_window(
            coordinates, window, order).astype(np.float64)
        self.local_tree = Rtree(scaled_coordinates, build_type='all')
        self.local_tree.set_order(
            order, order_required=fix_order, method=mode)
        self.local_tree.precalculate_powers()
        self._fit_settings = None

    @property
    def features(self):
        """int : number of data features (dimensions)"""
        return self._nfeatures

    @property
    def multi_set(self):
        """bool : True if solving for multiple data sets"""
        return self._multiset

    @property
    def n_sets(self):
        """int : The number of data sets to solve for"""
        return self._ndatasets

    @property
    def n_samples(self):
        """int : The number of samples in each data set"""
        return self._nsamples

    @property
    def window(self):
        """"""
        return self._radius.copy()

    @property
    def order(self):
        """"""
        return self.local_tree.order

    @property
    def fit_settings(self):
        return self._fit_settings

    @classmethod
    def _check_input_arrays(cls, coordinates, data, error=None, mask=None):
        """Checks the validity of arguments to __init__

        Checks that sample coordinates, values, error, and mask
        have compatible dimensions.  Also checks that there are
        enough samples overall to

        Raises a ValueError if an argument or parameter is not valid.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[None]
        if coordinates.ndim != 2:
            raise ValueError("coordinates array must have 1 (nsamples,) "
                             "or 2 (nfeatures, nsamples) features")

        ndata = coordinates.shape[-1]

        data = np.asarray(data, dtype=np.float64)
        shape = data.shape
        if shape[-1] != ndata:
            raise ValueError("data sample size does not match coordinates")
        if data.ndim not in [1, 2]:
            raise ValueError(
                "data must have 1 or 2 (multi-set) dimensions")

        if error is not None:
            error = np.asarray(error, dtype=np.float64)
            if error.shape != shape:
                raise ValueError("error shape does not match data")

        if mask is not None:
            mask = np.asarray(mask, dtype=np.bool)
            if mask.shape != shape:
                raise ValueError("mask shape does not match data")

        return coordinates, data, error, mask

    @classmethod
    def _check_order(cls, order, n_features, n_samples):
        order = np.asarray(order, dtype=np.int64)
        if order.ndim > 1:
            raise ValueError(
                "order should be a scalar or vector of 1 dimension")
        elif order.ndim == 1 and order.size != n_features:
            raise ValueError(
                "order vector does not match number of features")

        if order.ndim == 0:
            min_points = (order + 1) ** n_features
        else:
            min_points = np.product(order + 1)
        if n_samples < min_points:
            raise ValueError("too few data samples for order")

        return order

    def _process_input_data(self, data, error, mask,
                            negthresh=None, robust=None):
        """Formats the input data, error, and mask for subsequent use.

        Sets the data, mask, and error attributes to numpy.ndarrays
        of shape (n_sets, n_samples).

        The output mask will be a union of the input mask (if there
        is one) and finite data values and nonzero error values.
        If the user has provided `robust` or `negthresh` then the
        mask will be updated to reflect this.  See __init__ for
        further details.
        """
        data = np.asarray(data).astype(np.float64)

        if data.ndim == 2:
            self._multiset = True
            nsets = data.shape[0]
            if nsets > 1:
                self._ndatasets = nsets
            else:
                self._ndatasets = 1
        else:
            self._multiset = False
            self._ndatasets = 1
        data = np.atleast_2d(data)

        if mask is not None:
            mask = np.atleast_2d(np.asarray(mask).astype(bool))

        mask = robust_mask(data, robust, mask=mask, axis=-1)

        if negthresh is not None:
            invalid = np.logical_not(mask)
            data[invalid] = np.nan
            rms = bn.nanstd(data, ddof=1, axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mask &= data > (-rms * negthresh)

        invalid = np.logical_not(mask)

        doerror = error is not None
        if doerror:
            error = np.atleast_2d(np.asarray(error).astype(np.float64))
            self._error_weighted = True
            invalid |= ~np.isfinite(error)
            invalid |= error == 0
        else:
            error = np.ones_like(data)
            self._error_weighted = False

        error[invalid] = np.inf
        data[invalid] = 0.0

        self._valid_set = np.any(np.isfinite(error), axis=1)
        if not self._valid_set.any():
            raise ValueError("all data has been marked as invalid")

        self.data = data
        self.error = error
        self.mask = np.logical_not(invalid, out=mask)

    def _scale_to_window(self, coordinates, radius, order):
        if radius is None:
            radius = self.estimate_feature_windows(coordinates, order)
        else:
            radius = np.atleast_1d(radius)
            if radius.size < self._nfeatures:
                radius = np.full(self._nfeatures, radius[0])
        self._radius = radius.astype(np.float64)
        self._scale_offsets = coordinates.min(axis=1)

        return scale_coordinates(
            coordinates, self._radius, self._scale_offsets)

    @staticmethod
    def estimate_feature_windows(coordinates, order,
                                 feature_bins=10, percentile=50):
        """Estimates the fitting window for each feature"""
        x = np.asarray(coordinates).astype(float)
        scale = np.ptp(x, axis=1)
        x /= scale[:, None]
        x -= x.min(axis=1)[:, None]
        features, nsamples = x.shape

        feature_bins = np.max((int(feature_bins), 1))
        m = feature_bins if feature_bins == 1 else feature_bins - 1
        x0 = np.floor(x * m).astype(int)
        i = np.ravel_multi_index(
            x0, [feature_bins] * features, mode='raise')
        counts = np.bincount(i)

        bin_population = np.percentile(counts[counts > 0], percentile)
        o = np.asarray(order)
        if o.shape == ():
            o = np.full(features, int(o))
        # required samples for each local fit
        required_samples = np.product(o + 1)

        # The expected population if density was flat
        flat_population = bin_population * (feature_bins ** features)

        required_bins = flat_population / required_samples

        hyperside = 1 / (required_bins ** (1 / features)) * np.sqrt(features)
        return hyperside * scale

    def reduction_settings(self, smoothing, fit_threshold, cval,
                           edge_threshold, edge_algorithm,
                           relative_smooth, get_error):

        """Purely so this doesn't have to be repeated N times or passed
        in as ugly arguments"""
        distance_weighting = smoothing is not None
        error_weighting = self._error_weighted

        n_features = self.local_tree.features
        if distance_weighting:
            alpha = np.atleast_1d(np.asarray(smoothing, dtype=np.float64))
            if alpha.size not in [1, n_features]:
                raise ValueError(
                    "smoothing size does not match number of features")

            if not relative_smooth:
                if alpha.size != n_features:  # alpha size = 1
                    alpha = np.full(n_features, alpha[0])

                alpha *= alpha / (self.window ** 2)
                # alpha /= self.window  # get alpha in the same units as window

            if alpha.size == 1 or np.unique(alpha).size == 1:
                symmetrical_distance = True
                alpha = np.atleast_1d(np.float64(alpha[0]))
            else:
                symmetrical_distance = False

            asymmetrical_distance = not symmetrical_distance
        else:
            alpha = np.asarray([0.0])
            symmetrical_distance = False
            asymmetrical_distance = False

        order = self.local_tree.order
        if order is None:
            raise ValueError("Order has not been set on local tree")
        order_varies = self.local_tree.order_varies
        order_symmetry = self.local_tree.order_symmetry
        check_fit = True if fit_threshold else False
        if check_fit:
            fit_threshold = np.float64(fit_threshold)
        else:
            fit_threshold = np.float64(0.0)

        if edge_threshold is None or \
                (np.atleast_1d(edge_threshold) == 0).all():
            edge_threshold = 0.0
            check_edge = False
        else:
            check_edge = True

        edge_threshold = np.atleast_1d(edge_threshold).astype(np.float64)
        if edge_threshold.size != n_features:
            edge_threshold = np.full(n_features, edge_threshold[0])

        get_coordinates = asymmetrical_distance
        get_coordinates |= self.local_tree.order_method == 'edges'
        get_coordinates |= self.local_tree.order_method == 'extrapolate'
        get_coordinates |= check_edge

        get_mean = check_fit
        get_mean |= (order_varies and order_symmetry)
        get_mean |= get_error
        if order_symmetry:
            get_mean |= order == 0

        check_order0 = order_symmetry and order != 0 & order_varies

        if not check_order0 and order_symmetry:
            mean_only = order == 0
        else:
            mean_only = False

        if order_symmetry:
            order_minpoints = (order + 1) ** n_features
        else:
            order_minpoints = 1
            for i in range(n_features):
                order_minpoints *= order[i] + 1

        edge_func_lookup = {
            'com_distance': 1,
            'com_feature': 2,
            'range': 3,
        }
        edge_algorithm = str(edge_algorithm).lower().strip()
        edge_algorithm_idx = edge_func_lookup.get(edge_algorithm, 0)
        if edge_algorithm_idx == 0:
            edge_algorithm = 'none'

        order_algorithm = str(self.local_tree.order_method).lower().strip()
        order_func_lookup = {
            'edges': 1,
            'counts': 2,
            'extrapolate': 3
        }
        order_algorithm_idx = order_func_lookup.get(order_algorithm, 0)
        if order_algorithm_idx == 0:
            order_algorithm = 'none'

        self._fit_settings = {
            'n_features': n_features,
            'distance_weighting': distance_weighting,
            'error_weighting': error_weighting,
            'symmetrical_distance': symmetrical_distance,
            'asymmetrical_distance': asymmetrical_distance,
            'alpha': alpha,
            'order': np.atleast_1d(order),
            'order_varies': order_varies,
            'order_method': order_algorithm,
            'order_algorithm_idx': order_algorithm_idx,
            'order_required': self.local_tree.order_required,
            'order_symmetry': order_symmetry,
            'order_minpoints': order_minpoints,
            'check_fit': check_fit,
            'fit_threshold': fit_threshold,
            'get_coordinates': get_coordinates,
            'get_mean': get_mean,
            'mean_only': mean_only,
            'check_order0': check_order0,
            'cval': np.float64(cval),
            'check_edge': check_edge,
            'edge_threshold': edge_threshold,
            'edge_algorithm': edge_algorithm,
            'edge_algorithm_idx': edge_algorithm_idx
        }

        return self._fit_settings

    def _check_call_arguments(self, *args, smoothing=0.5,
                              edge_algorithm='com_distance',
                              edge_threshold=0.3):

        nargs = len(args)
        if len(args) not in [1, self.features]:
            raise ValueError(
                "%i-feature coordinates passed to %i-feature Resample"
                % (nargs, self.features))

        nsmooth = np.atleast_1d(smoothing).size
        if nsmooth != 1 and nsmooth != self.features:
            if nsmooth != self.features:
                raise ValueError(
                    "%i-feature smoothing passed to %i-feature Resample"
                    % (nsmooth, self.features))

        edge_algorithms = {'com_feature', 'com_distance', 'range'}
        if edge_algorithm not in edge_algorithms:
            raise ValueError("edge algorithm must be one of %s" %
                             edge_algorithms)

        for x in np.atleast_1d(edge_threshold):
            if x < 0 or x > 1:
                raise ValueError("edge threshold must be between 0 and 1")

    def __call__(self, *args, smoothing=0.0, relative_smooth=True,
                 fit_threshold=0.0, cval=np.nan,
                 edge_threshold=0.0, edge_algorithm='com_distance',
                 get_error=False, get_counts=False):

        self._check_call_arguments(*args, smoothing=smoothing,
                                   edge_algorithm=edge_algorithm,
                                   edge_threshold=edge_threshold)

        settings = self.reduction_settings(
            smoothing, fit_threshold, cval, edge_threshold, edge_algorithm,
            relative_smooth, get_error)

        visitor_grid = ResampleGrid(
            *args, tree_shape=self.local_tree.tree_shape,
            build_tree=True, scale_factor=self._radius,
            scale_offset=self._scale_offsets, dtype=np.float64)

        if settings['order_symmetry']:
            o = settings['order'][0]
        else:
            o = settings['order']

        visitor_tree = visitor_grid.tree
        visitor_tree.set_order(o, order_required=settings['order_required'],
                               method=settings['order_method'])
        visitor_tree.precalculate_powers()

        fit, error, counts = intersection_loop(
            self.data, self.error, self.mask, visitor_tree, self.local_tree,
            get_error, get_counts, settings)

        if not self.multi_set:
            fit = fit[0]
            if get_error:
                error = error[0]
            if get_counts:
                counts = counts[0]

        fit = visitor_grid.reshape_data(fit)
        if get_error or get_counts:
            result = (fit,)
            if get_error:
                result += (visitor_grid.reshape_data(error),)
            if get_counts:
                result += (visitor_grid.reshape_data(counts),)
            return result
        else:
            return fit


def intersection_loop(data, error, mask, visitor_tree, local_tree,
                      get_error, get_counts, settings):

    n_sets = data.shape[0]
    n_members = visitor_tree.n_members
    full_fit = np.full((n_sets, n_members), float(settings['cval']))
    if get_error:
        full_error = np.full((n_sets, n_members), 0.0)
    else:
        full_error = full_fit
    if get_counts:
        full_counts = np.full((n_sets, n_members), 0)
    else:
        full_counts = full_fit

    for block in range(visitor_tree.nblocks):
        if visitor_tree.block_population[block] == 0:
            continue
        if local_tree.hood_population[block] == 0:
            continue

        visitor_members, visitor_coordinates = visitor_tree.block_members(
            block, get_locations=True)

        local_members = local_tree.query_radius(
            visitor_coordinates, 1.0, return_distance=False)

        visitor_members, fit, fiterror, counts = tree_intersection(
            visitor_members, visitor_coordinates, visitor_tree.powers,
            local_members, local_tree.coordinates, local_tree.powers,
            data, error, mask, local_tree.power_order_idx,
            get_error, get_counts, settings)

        if visitor_members.size != 0:
            full_fit[:, visitor_members] = fit
            if get_error:
                full_error[:, visitor_members] = fiterror
            if get_counts:
                full_counts[:, visitor_members] = counts

    return full_fit, full_error, full_counts


def flatten_member_array(members):
    indices = np.empty(members.size + 1, dtype=np.int64)
    result = members[0]
    indices[0] = 0
    for i in range(members.size):
        result = np.hstack((result, members[i]))
        indices[i] = members[i].size
    indices[-1] = result.size
    return result, np.cumsum(indices)


def tree_intersection(visitor_members, visitor_coordinates, visitor_powers,
                      local_members, local_coordinates, local_powers,
                      data, error, mask, power_order_idx,
                      get_error, get_counts, settings):

    n_sets = data.shape[0]
    n_visitors = visitor_members.size
    fit_out = np.empty((n_sets, n_visitors))
    if get_error:
        error_out = np.empty((n_sets, n_visitors))
    else:
        error_out = fit_out
    if get_counts:
        counts_out = np.empty((n_sets, n_visitors), dtype=np.int64)
    else:
        counts_out = fit_out

    if n_sets == 0 or n_visitors == 0:
        return visitor_members, fit_out, error_out, counts_out

    error_weighting = settings['error_weighting']
    alpha = settings['alpha']
    order = settings['order']
    order_required = settings['order_required']
    order_varies = settings['order_varies']
    minimum_points = settings['order_minpoints']
    order_algorithm_idx = settings['order_algorithm_idx']
    fit_threshold = settings['fit_threshold']
    get_mean = settings['get_mean']
    mean_only = settings['mean_only']
    check_order0 = settings['check_order0']
    cval = settings['cval']
    edge_threshold = settings['edge_threshold']
    edge_algorithm_idx = settings['edge_algorithm_idx']

    if not error_weighting:
        error = np.ones((data.shape[0], 1))

    for visitor_id in prange(visitor_members.size):  # parallel if possible
        visitor = visitor_members[visitor_id]
        local_submembers = local_members[visitor_id]

        solve_visitor(
            data, mask, error, cval, get_error, get_counts,
            fit_out, error_out, counts_out,
            visitor_id, visitor,
            visitor_coordinates, visitor_powers,
            local_submembers, local_coordinates, local_powers,
            order, power_order_idx,
            order_algorithm_idx, order_varies,
            order_required, minimum_points, check_order0,
            edge_algorithm_idx, edge_threshold, alpha,
            fit_threshold, mean_only, get_mean)

    return visitor_members, fit_out, error_out, counts_out


def resample(coordinates, data, *locations, error=None, mask=None,
             window=None, order=1, mode='edges', fix_order=True,
             smoothing=0.0, relative_smooth=True,
             robust=None, negthresh=None, fit_threshold=0.0, cval=np.nan,
             get_error=False, get_counts=False):

    resampler = Resample(coordinates, data, error=error, mask=mask,
                         order=order, window=window, fix_order=fix_order,
                         robust=robust, negthresh=negthresh, mode=mode)

    return resampler(*locations, smoothing=smoothing,
                     relative_smooth=relative_smooth,
                     fit_threshold=fit_threshold, cval=cval,
                     get_error=get_error, get_counts=get_counts)
