import cProfile
from abc import ABC, abstractmethod
import collections
import warnings
import bitarray
import numpy as np
import pandas as pd

from DiverseSelector import predict_radius, KDTree, GridPartitioning, OptiSim, MaxSum, MaxMin
from multiprocessing.dummy import Pool


class SelectionBase(ABC):
    """Base class for subset selection."""

    def select(self, arr, num_selected, labels=None):
        """
         Algorithm for selecting points.

        Parameters
        ----------
        arr: np.ndarray
            Array of features if fun_distance is provided.
            Otherwise, treated as distance matrix.
        num_selected: int
            Number of points that need to be selected
        labels: np.ndarray
            Labels for performing algorithm withing clusters.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if labels is None:
            return self.select_from_cluster(arr, num_selected)

        # compute the number of samples (i.e. population or pop) in each cluster
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        pop_clusters = {unique_label: len(np.where(labels == unique_label)[0])
                        for unique_label in unique_labels}
        # compute number of samples to be selected from each cluster
        n = num_selected // num_clusters

        # update number of samples to select from each cluster based on the cluster population.
        # this is needed when some clusters do not have enough samples in them (pop < n) and
        # needs to be done iteratively until all remaining clusters have at least n samples
        selected_ids = []
        while np.any([value <= n for value in pop_clusters.values() if value != 0]):
            for unique_label in unique_labels:
                if pop_clusters[unique_label] != 0:
                    # get index of sample labelled with unique_label
                    cluster_ids = np.where(labels == unique_label)[0]
                    if len(cluster_ids) <= n:
                        # all samples in the cluster are selected & population becomes zero
                        selected_ids.append(cluster_ids)
                        pop_clusters[unique_label] = 0
            # update number of samples to be selected from each cluster
            totally_used_clusters = list(pop_clusters.values()).count(0)
            n = (num_selected - len(np.hstack(selected_ids))) // \
                (num_clusters - totally_used_clusters)

            warnings.warn(
                f"Number of molecules in one cluster is less than"
                f" {num_selected}/{num_clusters}.\nNumber of selected "
                f"molecules might be less than desired.\nIn order to avoid this "
                f"problem. Try to use less number of clusters"
            )

        for unique_label in unique_labels:
            if pop_clusters[unique_label] != 0:
                # sample n ids from cluster labeled unique_label
                cluster_ids = np.where(labels == unique_label)[0]
                selected = self.select_from_cluster(arr, n, cluster_ids)
                selected_ids.append(cluster_ids[selected])

        return np.hstack(selected_ids).flatten().tolist()

    @abstractmethod
    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.array


        Returns
        -------
        selected: list
            List of ids of molecules that are belonged to the one cluster

        """
        pass


# class KDTreeBase(SelectionBase, ABC):
#     """Base class for KDTree based subset selection.
#
#     Adapted from https://johnlekberg.com/blog/2020-04-17-kd-tree.html
#     """
#
#     def __int__(self):
#         """Initializing class."""
#         self.func_distance = lambda x, y: np.linalg.norm(x - y)
#         self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])
#         self.NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])








class DirectedSphereExclusion(SelectionBase):
    """Selecting points using Directed Sphere Exclusion algorithm.

    Starting point is chosen as the reference point and not included in the selected molecules. The
    distance of each point is calculated to the reference point and the points are then sorted based
    on the ascending order of distances. The points are then evaluated in their sorted order, and
    are selected if their distance to all the other selected points is at least r away. Euclidian
    distance is used by default and the r value is automatically generated if not passed to satisfy
    the number of molecules requested.

    Adapted from https://doi.org/10.1021/ci025554v
    """

    def __init__(self, r=None, tolerance=5.0, func_distance=lambda x, y: np.linalg.norm(x - y),
                 start_id=0, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        r: float
            Initial guess of radius for directed sphere exclusion algorithm, no points within r
            distance to an already selected point can be selected.
        tolerance: float
            Percentage error of number of molecules actually selected from number of molecules
            requested.
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        start_id: int
            Index for the first point to be selected.
        random_seed: int
            Seed for random selection of points be evaluated.
        """
        self.r = r
        self.tolerance = tolerance
        self.func_distance = func_distance
        self.starting_idx = start_id
        self.random_seed = random_seed
        self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])

    def algorithm(self, arr):
        """
        Directed sphere exclusion algorithm logic.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        selected = []
        candidates = np.delete(np.arange(0, len(arr)), self.starting_idx)
        distances = []
        ref_point = arr[self.starting_idx]

        # pool = Pool()
        # distances = list(pool.map(lambda x: (self.func_distance(ref_point, arr[x]), x),
        #                      candidates))
        for idx in candidates:
            ref_point = arr[self.starting_idx]
            data_point = arr[idx]
            distance = self.func_distance(ref_point, data_point)
            distances.append((distance, idx))
        distances.sort()
        order = [idx for dist, idx in distances]


        kdtree = self._kdtree(arr)
        bv = bitarray.bitarray(len(arr))
        bv[:] = 0
        bv[self.starting_idx] = 1

        for idx in order:
            if not bv[idx]:
                selected.append(idx)
                elim = self._find_nearest_neighbor(kdtree=kdtree, point=arr[idx], threshold=self.r)
                for index in elim:
                    bv[index] = 1

        return selected

    def _kdtree(self, arr):
        """Construct a k-d tree from an iterable of points.

        Parameters
        ----------
        arr: list or np.ndarray
            Coordinate array of points.

        Returns
        -------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        """

        k = len(arr[0])

        def build(points, depth, old_indices=None):
            """Build a k-d tree from a set of points at a given depth."""
            if len(points) == 0:
                return None
            middle = len(points) // 2
            indices, points = zip(*sorted(enumerate(points), key=lambda x: x[1][depth % k]))
            if old_indices is not None:
                indices = [old_indices[i] for i in indices]
            return self.BT(
                value=points[middle],
                index=indices[middle],
                left=build(
                    points=points[:middle],
                    depth=depth + 1,
                    old_indices=indices[:middle],
                ),
                right=build(
                    points=points[middle + 1:],
                    depth=depth + 1,
                    old_indices=indices[middle + 1:],
                ),
            )

        kdtree = build(points=arr, depth=0)
        return kdtree

    def _find_nearest_neighbor(self, kdtree, point, threshold, sort=True):
        """
        Find the nearest neighbors in a k-d tree for a point.

        Parameters
        ----------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Query point for search.
        threshold: float
            The boundary used to mark all the points whose distance is within the threshold.
        sort: boolean
            Whether the results should be sorted based on lowest distance or not.

        Returns
        -------
        to_eliminate: list
            A list containing all the indices of points too close to the newly selected point.
        """
        k = len(point)
        to_eliminate = []

        def search(tree, depth):
            # Recursively search through the k-d tree to find the
            # nearest neighbor.

            if tree is None:
                return

            distance = self.func_distance(tree.value, point)
            if distance < threshold:
                to_eliminate.append((distance, tree.index))

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=close, depth=depth + 1)
            if diff < threshold:
                search(tree=away, depth=depth + 1)

        search(tree=kdtree, depth=0)
        to_eliminate = [index for dist, index in to_eliminate]
        if sort:
            to_eliminate.sort()
        return to_eliminate

    def _nearest_neighbor(self, kdtree, point):
        """
        Find the nearest neighbors in a k-d tree for a point.

        Parameters
        ----------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Query point for search.
        threshold: float
            The boundary used to mark all the points whose distance is within the threshold.

        Returns
        -------
        to_eliminate: list
            A list containing all the indices of points too close to the newly selected point.
        """
        k = len(point)
        best = None

        def search(tree, depth):
            # Recursively search through the k-d tree to find the
            # nearest neighbor.
            nonlocal best

            if tree is None:
                return

            distance = self.func_distance(tree.value, point)
            if best is None or distance < best.distance:
                best = self.NNRecord(point=tree.value, distance=distance)

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=close, depth=depth + 1)
            if diff < best.distance:
                search(tree=away, depth=depth + 1)

        search(tree=kdtree, depth=0)
        return best

    def _eliminate(self, tree, point, threshold, num_eliminate, bv):
        """Eliminates points from being selected in future rounds.

        Parameters
        ----------
        tree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Point where close neighbors should be eliminated.
        threshold: float
            An average of all the furthest distances found using find_furthest_neighbor
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        bv: bitarray
            Bitvector marking picked/eliminated points.

        Returns
        -------
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        """
        elim_candidates = self._find_nearest_neighbor(tree, point, threshold)
        elim_candidates = elim_candidates[:self.ratio]
        num_eliminate -= len(elim_candidates)
        if num_eliminate < 0:
            elim_candidates = elim_candidates[:num_eliminate]
        for index in elim_candidates:
            bv[index] = 1
        return num_eliminate

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm that uses sphere_exclusion for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        return predict_radius(self, arr, num_selected, cluster_ids)


from sklearn.metrics import pairwise_distances

df = lambda x, y: pairwise_distances([x, y])[0, 1]

x = df([0,0], [0,1])
print(x)


func = pairwise_distances
selectors = [MaxMin(func_distance=func),
             MaxSum(func_distance=func),
             OptiSim(),
             DirectedSphereExclusion(),
             GridPartitioning(cells=10),
             KDTree()]

def get_array(string):
    return np.array([int(elem) for elem in string])
#
# data_1024_all = pd.read_excel('BBB_SECFP6_1024.xlsx')
# data_2048_all = pd.read_excel('BBB_SECFP6_2048.xlsx')
# data_1024 = np.vstack(pd.read_excel('BBB_SECFP6_1024.xlsx').fingerprint.apply(get_array).values)
# data_2048 = np.vstack(pd.read_excel('BBB_SECFP6_2048.xlsx').fingerprint.apply(get_array).values)
#
# print(data_1024.shape, data_2048.shape)
#
# num_selected = 100
#
# data_1024 = data_1024[:, :2]
# # print(tester.shape)
# # print(pairwise_distances([tester[0], tester[1]])[0,1])








arr_size = 10000
num_selected = 1000


import random
random_point = lambda: [random.random(), random.random()]
reference_points = [ random_point() for _ in range(arr_size) ]
# reference_points = [[1, 2], [3, 2], [4, 1], [3, 5], [1, 7], [3, 8], [9, 5], [4, 9], [11, 4], [5, 1], [7, 3], [8, 9], [10, 1], [3, 3]]
# 2048
# def test():
    # for i in range(10):

data_1024 = np.array(reference_points)




# cProfile.run(f"MaxMin(lambda x: pairwise_distances(x, metric='euclidean')).select(arr=tester, num_selected={num_selected})")
cProfile.run(f"DirectedSphereExclusion(tolerance=10000000000000, r=0.065).select(arr=data_1024, num_selected={num_selected})")



ref_point = data_1024[0]
order = []
for idx, point in enumerate(data_1024):
    dist = np.linalg.norm(ref_point-point)
    order.append((dist, idx))
order.sort()
order = [idx for dist, idx in order]
bv = np.zeros(len(data_1024))
bv[0] = 1
selected = []
from scipy import spatial
tree = spatial.KDTree(data_1024)
print(tree)
for idx in order:
    if bv[idx]:
        continue
    bv[idx] = 1
    selected.append(idx)
    x = tree.query_ball_point(data_1024[idx], 0.0000005, workers=-1)
    for i in x:
        bv[i] = 1
print(selected)
print(len(selected))





ref_point = data_1024[0]
order = []
for idx, point in enumerate(data_1024):
    dist = np.linalg.norm(ref_point-point)
    order.append((dist, idx))
order.sort()
order = [idx for dist, idx in order]
bv = np.zeros(len(data_1024))
bv[0] = 1
selected = []
from scipy import spatial
tree = spatial.KDTree(data_1024)
query = tree.query_ball_tree(tree, 0.005)
for idx in order:
    if bv[idx]:
        continue
    bv[idx] = 1
    selected.append(idx)
    x = query[idx]
    for i in x:
        bv[i] = 1
print(selected)
print(len(selected))

#[3218, 4565, 3385, 811, 2728, 1094, 5349, 2, 766, 884, 3041, 3280, 1011, 1538, 4298, 4560, 5158, 640, 1296, 3427, 5123, 1261, 3919, 542, 2781, 4997, 5344, 7200, 904, 3420, 3436, 5059, 8, 2518, 2673, 2771, 5095, 706, 5243, 5729, 737, 1531, 1602, 1724, 1729, 3251, 3357, 3394, 4350, 5256, 6164, 6430, 142, 435, 1088, 3182, 3250, 5547, 488, 2288, 3107, 3152, 3857, 4456, 6567, 2206, 4530, 6714, 3353, 3354, 4621, 4646, 5174, 6949, 2975, 3355, 3897, 4518, 5173, 6482, 64, 758, 847, 863, 1070, 1978, 4850, 5126, 5136, 5259, 6749, 6865, 244, 318, 2883, 4453, 4727, 4917, 6489, 6897, 7308, 58, 406, 984, 2412, 2637, 2665, 2898, 3140, 3434, 4947, 5048, 5116, 5447, 5896, 6672, 1378, 1709, 2165, 3002, 3142, 3404, 4457, 4511, 4632, 4729, 5065, 5197, 5860, 6163, 6227, 6585, 78, 150, 2350, 2905, 3191, 3196, 4647, 4869, 5194, 5210, 7252, 87, 443, 3336, 4383, 6469, 7004, 1939, 2125, 5584, 5715, 6167, 6209, 71, 553, 808, 3285, 5004, 5151, 5156, 5215, 4493, 4505, 4612, 5190, 6085, 6237, 59, 830, 2387, 2523, 2831, 5001, 5208, 5743, 6540, 77, 401, 733, 1215, 4966, 4980, 5199, 6546, 7224, 1199, 1343, 2502, 3223, 3431, 3944, 4484, 4629, 4789, 4845, 4979, 5133, 5175, 5289, 6843, 7173, 242, 450, 584, 4196, 6158, 6938, 9, 2456, 3841, 4120, 4625, 4972, 456, 6045, 6858, 7010, 554, 1849, 4209, 4770, 4940, 4951, 5149, 5258, 5555, 6346, 1855, 4993, 5029, 6310, 7238, 7329, 1753, 3533, 4927, 5051, 5063, 5075, 6166, 7289, 441, 3132, 5817, 2448, 3236, 3915, 4428, 4481, 4945, 4968, 5253, 1722, 3753, 4939, 4963, 5131, 5218, 2199, 3242, 5096, 5120, 5127, 6467, 6764, 3688, 4464, 4613, 5601, 4918, 4919, 6418, 1154, 5043, 6696, 1863, 4502, 5017, 5121, 5143, 5580, 5625, 6195, 4278, 4889, 6095, 672, 780, 1857, 2516, 3726, 4599, 4950, 5110, 339, 1311, 1852, 5394, 6756, 334, 3264, 3359, 3767, 4716, 5132, 6901, 1754, 155, 4182, 2151, 3581, 5155, 5183, 6086, 6564, 4095, 4762, 5009, 1141, 3525, 4811, 4887, 433, 3598, 4177, 6243, 6533, 5044, 5370, 6730, 4909, 5027, 1209, 2062, 5918, 6684, 591, 1143, 4287, 5671, 2786, 4913, 5788, 5794, 4995, 6055, 2670, 3828, 4894, 6750, 2390, 5318, 5374, 2333, 4024, 6239, 4916, 5837, 6994, 6566, 1188, 2106, 4495, 5047, 3839, 4029, 4330, 872, 6753, 4985, 4908, 3700, 5005, 5576]