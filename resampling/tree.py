import itertools
from numba import jit, njit, prange
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree
from .resample_utils import power_product, cull_members, polynomial_exponents

_fast_flags = {'nsz', 'nnan', 'ninf'}


class Rtree(object):

    def __init__(self, argument, shape=None, build_type='all'):

        self._shape = None
        self._nfeatures = None
        self._nblocks = None
        self._multipliers = None
        self._deltas = None
        self._search_sides = None
        self._tree = None
        self._balltree = None
        self._ball_initialized = False
        self._hood_initialized = False
        self._order = None
        self._order_symmetry = None
        self._order_required = None
        self._order_varies = None
        self._order_method = None
        self._powers_precalculated = False
        self.block_offsets = None
        self.block_population = None
        self.populated = None
        self.hood_population = None
        self.coordinates = None
        self.powers = None
        self.power_order_idx = None

        arg = np.asarray(argument)
        if np.asarray(arg).ndim > 1:
            self.build_tree(arg, shape=shape, method=build_type)
        else:
            self._set_shape(tuple(arg))

    @property
    def tree_shape(self):
        return tuple(self._shape)

    @property
    def features(self):
        return self._nfeatures

    @property
    def order(self):
        return self._order

    @property
    def order_symmetry(self):
        return self._order_symmetry

    @property
    def order_required(self):
        return self._order_required

    @property
    def order_varies(self):
        return self._order_varies

    @property
    def order_method(self):
        return self._order_method

    @property
    def powers_precalculated(self):
        return self._powers_precalculated

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def deltas(self):
        return self._deltas

    @property
    def multipliers(self):
        return self._multipliers

    @property
    def reverse_deltas(self):
        return self._reverse_deltas

    @property
    def search_sides(self):
        return self._search_sides

    @property
    def n_members(self):
        if self.coordinates is None:
            return 0
        else:
            return self.coordinates.shape[1]

    def _set_shape(self, shape):
        self._shape = np.asarray(shape).ravel().astype(int)
        self._nfeatures = len(self._shape)
        self._nblocks = np.prod(self._shape)
        searches = np.array(
            list(itertools.product(*([[-1, 0, 1]] * self._nfeatures))))
        self._multipliers = abs(searches)
        self._deltas = -1 * (searches < 0)
        reverse_search = searches * -1
        self._reverse_deltas = -1 * (reverse_search < 0)
        self._search_sides = searches.T

        self._tree = None
        self.block_offsets = None
        self.block_population = None
        self.populated = None
        self.hood_population = None
        self.power_order_idx = None

    def to_index(self, values):
        c = np.asarray(values, dtype=int)
        single = c.ndim == 1
        if single:
            c = c[:, None]
        index = np.ravel_multi_index(c, self._shape, mode='clip')
        return index[0] if single else index

    def from_index(self, index):
        c = np.asarray(index, dtype=int)
        shape = c.shape
        single = shape == ()
        index = np.unravel_index(np.atleast_1d(c).ravel(), self._shape)
        output = np.empty((self._nfeatures, c.size), dtype=int)
        output[:] = index
        return output.ravel() if single else output

    def build_tree(self, coordinates, shape=None, method='all'):
        if shape is None:
            self._set_shape(coordinates.astype(int).max(axis=1) + 1)
        else:
            self._set_shape(shape)
        mstr = str(method).lower().strip()
        self.coordinates = np.asarray(coordinates).astype(float)
        if mstr == 'hood':
            self._build_hood_tree()
        elif mstr == 'balltree':
            self._build_ball_tree()
        elif mstr == 'none':
            pass
        else:
            self._build_hood_tree()
            self._build_ball_tree()

    def _build_ball_tree(self):
        self._ball_initialized = False
        self._balltree = BallTree(self.coordinates.T)
        self._ball_initialized = True

    def query_radius(self, coordinates, radius=1.0, **kwargs):
        if not self._ball_initialized:
            raise RuntimeError("Ball tree not initialized")
        if coordinates.ndim == 1:
            c = coordinates[None]
        else:
            c = coordinates.T
        return self._balltree.query_radius(c, radius, **kwargs)

    def _build_hood_tree(self):
        self._hood_initialized = False
        bins = self.to_index(self.coordinates)
        inds = np.arange(bins.size)
        nbins = np.prod(self._shape)
        self._tree = csr_matrix((inds, [bins, inds]),
                                shape=(nbins, bins.size))
        self._tree = np.split(self._tree.data, self._tree.indptr[1:-1])

        self.block_offsets = self.coordinates - self.coordinates.astype(int)
        self.block_population = np.array([s.size for s in self._tree])
        self.populated = np.nonzero(self.block_population > 0)[0]
        self.hood_population = np.zeros_like(self.block_population)
        self.max_in_hood = np.zeros_like(self.block_population)
        for block in self.populated:
            hoods = self.neighborhood(block)
            self.hood_population[block] = np.sum(
                self.block_population[hoods])
            self.max_in_hood[block] = np.max(
                self.block_population[hoods])
        self._hood_initialized = True

    def block_members(self, block, get_locations=False):
        if not self._hood_initialized:
            raise RuntimeError("neighborhood tree not initialized")
        members = self._tree[block]
        if not get_locations:
            return members
        return members, self.coordinates[:, members]

    def neighborhood(self, index, cull=False, cull_idx=False):
        expanded = self.from_index(index)[:, None] + self._search_sides
        bad = np.any(
            (expanded < 0) | (expanded >= self._shape[:, None]), axis=0)
        hood = self.to_index(expanded)
        if cull:
            keep = ~bad
            hood = hood[keep]
            return (hood, keep) if cull_idx else hood
        else:
            hood[bad] = -1
            return hood

    def hood_members(self, center_block, get_locations=False):
        if not self._hood_initialized:
            raise RuntimeError("neighborhood tree not initialized")
        hood, populated = self.neighborhood(
            center_block, cull=True, cull_idx=True)
        if hood.size == 0:
            return (hood, []) if get_locations else hood

        members = np.empty(self.block_population[hood].sum(), dtype=int)
        mults = self._multipliers[populated]
        deltas = self._deltas[populated]

        population = 0
        for block, m, d in zip(hood, mults, deltas):

            valid_members = cull_members(
                self.block_offsets, self._tree[block], m, d, 0)
            new_population = population + valid_members.size
            members[population:new_population] = valid_members
            population = new_population

        members = members[:population]
        if get_locations:
            return members, self.coordinates[:, members]
        else:
            return members

    def set_order(self, order, order_required=True, method='counts'):

        method = str(method).lower()
        if method not in ['counts', 'edges', 'extrapolate']:
            raise ValueError("unknown order method: %s" % method)

        o = np.asarray(order)
        order_symmetry = o.shape == ()
        if not order_symmetry:
            if order.size != self._nfeatures:
                raise ValueError(
                    "asymmetrical order does not match features")
        if order_symmetry:
            self._order = int(order)
        else:
            self._order = o.astype(int)
        self._order_symmetry = order_symmetry
        self._order_required = order_required
        self._order_method = method
        self._order_varies = not order_required and order_symmetry
        # Note, it is possible to vary orders if order_symmetry is False,
        # but will require either a ton of either memory or computation time.
        # (plus I'll need to write the code)

    def precalculate_powers(self):

        self._powers_precalculated = False
        self.powers = None

        self.power_order_idx = np.full(np.max(self._order) + 2, -1)
        if self._order_varies:
            order_set = False
            self.power_order_idx[0] = 0
            ocount = 0
            for o in range(self._order + 1):
                exponents = polynomial_exponents(o, ndim=self._nfeatures)
                p = power_product(self.coordinates, exponents)
                ocount += p.shape[0]
                self.power_order_idx[o + 1] = ocount
                if not order_set:
                    self.powers = p
                    order_set = True
                else:
                    self.powers = np.vstack([self.powers, p])
        else:
            exponents = polynomial_exponents(self._order, ndim=self._nfeatures)
            o = np.max(self._order)
            self.powers = power_product(self.coordinates, exponents)
            self.power_order_idx[o] = 0
            self.power_order_idx[o + 1] = self.powers.shape[0]

        self._powers_precalculated = True

    def __call__(self, x, reverse=False):
        if reverse:
            return self.from_index(x)
        else:
            return self.to_index(x)


# The following functions are not used but available as an
# alternative to BallTree.
@njit(cache=True, fastmath=_fast_flags)
def update_cross_distance(start, visitor_members, local_members, separations,
                          out_visit, out_local, out_separation):

    nvisitor = visitor_members.size
    nlocal = local_members.size
    n = start
    for i in range(nvisitor):
        visitor = visitor_members[i]
        for j in range(nlocal):
            d = separations[i, j]
            if d <= 1.0:
                out_visit[n] = visitor
                out_local[n] = local_members[j]
                out_separation[n] = d
                n += 1
    return n


@jit(cache=True, fastmath=_fast_flags)
def block_intersection(block, visitor_tree, local_tree,
                       get_distance=True, get_edges=True):

    n_visitors = visitor_tree.block_population[block]
    if n_visitors == 0:
        return [], []
    visitor_members, visitor_locations = visitor_tree.block_members(
        block, get_locations=True)
    visitor_offsets = visitor_tree.block_offsets

    hood_population = local_tree.hood_population[block]
    if hood_population == 0:
        return [], []

    hoods, populated = local_tree.neighborhood(
        block, cull=True, cull_idx=True)
    popidx = np.nonzero(populated)[0]
    hood_members = [local_tree.block_members(hood) for hood in hoods]
    visitor_deltas = visitor_tree.deltas[popidx]
    local_deltas = local_tree.reverse_deltas[popidx]
    multipliers = visitor_tree.multipliers[popidx]  # same for both

    return hood_loop(
        hoods, hood_population, block,
        hood_members, visitor_members,
        local_tree.block_offsets, visitor_offsets,
        visitor_tree.coordinates, local_tree.coordinates,
        multipliers, visitor_deltas, local_deltas,
        get_edges, get_distance)


@jit(parallel=False, cache=True, fastmath=_fast_flags)
def hood_loop(hoods, hood_population, block,  # max_populatons (before block)
              hood_members, visitor_members,
              localblock_offsets, visitor_offsets,
              visitor_coordinates, local_coordinates,
              multipliers, visitor_deltas, local_deltas,
              get_edges, get_distance):

    nhoods = hoods.size
    n_visitors = visitor_members.size
    all_visitor_ids = np.arange(n_visitors)

    if get_edges:
        ndim = visitor_deltas.shape[1]
        hood_left = np.zeros((ndim, n_visitors, hood_population), dtype=int)
    else:
        hood_left = None

    if get_distance:
        hood_dist = np.empty((n_visitors, hood_population))
    else:
        hood_dist = None

    hood_counts = np.zeros(n_visitors, dtype=int)
    hood_founds = np.empty((n_visitors, hood_population), dtype=int)

    for i in range(nhoods):
        local_hood = hoods[i]
        in_the_hood = local_hood == block
        local_members = hood_members[i]
        if local_members.size == 0:
            continue

        if in_the_hood:
            locals_near_block = local_members
            off_center = None
        else:
            off_center = multipliers[i]
            locals_near_block = cull_members(
                localblock_offsets, local_members,
                off_center, visitor_deltas[i], 0)
        if locals_near_block.size == 0:
            continue

        if in_the_hood:
            visitor_ids = all_visitor_ids
            visitors_near_hood = visitor_members
        else:
            visitor_ids = cull_members(
                visitor_offsets, visitor_members, off_center,
                local_deltas[i], 1)
            if visitor_ids.size == 0:
                continue
            visitors_near_hood = visitor_members[visitor_ids]

        visitor_locations = visitor_coordinates[:, visitors_near_hood]
        filter_coordinates(
            visitor_locations, local_coordinates,  # coordinates
            visitor_ids, locals_near_block,  # reduced and full indices
            hood_founds, hood_counts,  # outputs
            hood_dist, hood_left)

    return hood_founds, hood_counts, hood_dist, hood_left


@njit(cache=True, fastmath=_fast_flags)
def filter_coordinates(visitor_locations,
                       local_coordinates,
                       visitor_ids,
                       locals_near_block,
                       hood_found,
                       hood_count,
                       hood_dist,
                       hood_left):

    ndim, nupdate = visitor_locations.shape
    nsamples = locals_near_block.size
    doedge = hood_left is not None
    dodistance = hood_dist is not None
    for i in prange(nupdate):
        ind0 = visitor_ids[i]
        for j in range(nsamples):
            idx = hood_count[ind0]
            ind1 = locals_near_block[j]
            d2 = 0.0
            for k in range(ndim):
                diff = visitor_locations[k, i] - local_coordinates[k, ind1]
                if doedge:
                    if diff < 0:
                        hood_left[k, ind0, idx] += 1
                diff *= diff
                d2 += diff
                if d2 > 1:
                    break
            else:
                hood_count[ind0] += 1
                hood_found[ind0, idx] = ind1
                if dodistance:
                    hood_dist[ind0, idx] = d2
