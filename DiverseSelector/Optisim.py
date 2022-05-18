import numpy as np
from sklearn.datasets import make_blobs


class OptiSim:
    def __init__(self, r=None, k=10):
        self.r = r
        self.k = k
        self.random_seed = 42
        self.starting_idx = 0

    def optisim(self, arr, labels=None):
        selected = [self.starting_idx]
        recycling = []

        candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
        subsample = {}
        while len(candidates) > 0:
            while len(subsample) < self.k:
                if len(candidates) == 0:
                    if len(subsample) > 0:
                        break
                    return selected
                rng = np.random.default_rng(seed=self.random_seed)
                random_int = rng.integers(low=0, high=len(candidates), size=1)[0]
                index_new = candidates[random_int]
                distances = []
                for selected_idx in selected:
                    data_point = arr[index_new]
                    selected_point = arr[selected_idx]
                    distance_sq = 0
                    for i, point in enumerate(data_point):
                        distance_sq += (selected_point[i] - point) ** 2
                    distances.append(np.sqrt(distance_sq))
                min_dist = min(distances)
                if min_dist > self.r:
                    subsample[index_new] = min_dist
                else:
                    recycling.append(index_new)
                candidates = np.delete(np.arange(0, len(arr)),
                                       selected + recycling + list(subsample.keys()))
            selected.append(max(zip(subsample.values(), subsample.keys()))[1])
            candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
            subsample = {}

        return selected

    def select_from_cluster(self, arr, num_select, indices=None):
        if indices is not None:
            arr = arr[indices]
        if self.r is None:
            # Use numpy.optimize.bisect instead
            arr_range = (max(arr[:, 0]) - min(arr[:, 0]),
                         max(arr[:, 1]) - min(arr[:, 1]))
            rg = max(arr_range) / num_select * 3
            self.r = rg
            result = self.optisim(arr)
            if len(result) == num_select:
                return result

            low = rg if len(result) > num_select else 0
            high = rg if low == 0 else None
            bounds = [low, high]
            count = 0
            while len(result) != num_select and count < 20:
                if bounds[1] is None:
                    rg = bounds[0] * 2
                else:
                    rg = (bounds[0] + bounds[1]) / 2
                self.r = rg
                result = self.optisim(arr)
                if len(result) > num_select:
                    bounds[0] = rg
                else:
                    bounds[1] = rg
                count += 1
            self.r = None
            return result
        else:
            return self.optisim(arr)

    def select(self, arr, num_selected, func_distance=None, labels=None):
        """
         MinMax algorithm for selecting points.

        Parameters
        ----------
        arr: np.ndarray
            Array of features if fun_distance is provided.
            Otherwise, treated as distance matrix.
        func_distance: callable
            function for calculating the pairwise distance between instances of the array.
            Default is None.
        num_selected: int
            number of points that need to be selected
        labels: np.ndarray
            labels for performing algorithm withing clusters.
            Default is None.

        Returns
        -------
        selected: list
            list of ids of selected molecules
        """
        self.n_mols = arr.shape[0]
        if func_distance is not None:
            self.arr_dist = func_distance(arr)
        else:
            self.arr_dist = arr

        if labels is not None:
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels)
            selected_all = []
            totally_used = []

            amount_molecules = np.array(
                [len(np.where(labels == unique_label)[0]) for unique_label in unique_labels])

            n = (num_selected - len(selected_all)) // (num_clusters - len(totally_used))

            while np.any(amount_molecules <= n):
                for unique_label in unique_labels:
                    if unique_label not in totally_used:
                        cluster_ids = np.where(labels == unique_label)[0]
                        if len(cluster_ids) <= n:
                            selected_all.append(cluster_ids)
                            totally_used.append(unique_label)

                n = (num_selected - len(selected_all)) // (num_clusters - len(totally_used))
                amount_molecules = np.delete(amount_molecules, totally_used)

                warnings.warn(f"Number of molecules in one cluster is less than"
                              f" {num_selected}/{num_clusters}.\nNumber of selected "
                              f"molecules might be less than desired.\nIn order to avoid this "
                              f"problem. Try to use less number of clusters")

            for unique_label in unique_labels:
                if unique_label not in totally_used:
                    cluster_ids = np.where(labels == unique_label)[0]
                    print(cluster_ids.shape)
                    # arr_dist_cluster = self.arr_dist[cluster_ids]
                    # selected = self.select_from_cluster(arr_dist_cluster, n)
                    selected = self.select_from_cluster(self.arr_dist, n, cluster_ids)
                    selected_all.append(cluster_ids[selected])
            return np.hstack(selected_all).flatten().tolist()
        else:
            selected = self.select_from_cluster(self.arr_dist, num_selected)
            return selected

# if __name__ == "__main__":
#     arr, labels = make_blobs(n_samples=100, n_features=2, centers=1, random_state=42)
#     selected = OptiSim(k=10).select(arr, 12)
#     print(selected)
