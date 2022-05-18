from DiverseSelector.base import SelectionBase
import numpy as np

class OptiSim(SelectionBase):
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
