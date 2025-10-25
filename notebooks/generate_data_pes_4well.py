#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from selector import MaxMin, MaxSum, OptiSim, DISE, Medoid, GridPartition
from selector.measures.diversity import logdet, wdud

"""
python generate_data_pes_4well.py generate_data

python generate_data_pes_4well.py select_subsets data/monte_carlo_k*/dataset_*.json

python generate_data_pes_4well.py compute_diversity data/monte_carlo_k02/dataset_1.0e+07_k02_size10000.json

"""

class FourWellPotential:
    def __init__(self):
        # papers taken from paper
        self.v0 = 5.0
        self.a0 = 0.6
        self.a = [3.0, 1.5, 3.2, 2.0]
        self.b_q1 = 0.1
        self.b_q2 = 0.1
        self.s_q1 = [0.3, 1.0, 0.4, 1.0]
        self.s_q2 = [0.4, 1.0, 1.0, 0.1]
        self.alpha = [1.3, -1.5, 1.4, -1.3]
        self.beta = [-1.6, -1.7, 1.8, 1.23]
        # evaluate potential at A, B, C, D minima (taken from paper)
        points = np.array([[1.29, -1.65], [1.4, 1.78], [-1.29, -1.53], [-1.17, 1.56]])
        self.minima = [points, self.evaluate(points)]

    def evaluate(self, points):
        if points.ndim != 2 and points.shape[1] != 2:
            raise ValueError("Argument points must be 2D array with 2 columns.")

        # TODO: This needs to be vectorized
        values = []
        for q1, q2 in points:
            v = self.v0 + self.a0 * (np.exp(-((q1 - self.b_q1) ** 2) - (q2 - self.b_q2) ** 2))
            for i in range(4):
                v -= self.a[i] * np.exp(
                    -self.s_q1[i] * (q1 - self.alpha[i]) ** 2
                    - self.s_q2[i] * (q2 - self.beta[i]) ** 2
                )
            values.append(v)
        return np.array(values)

    def generate_points_uniform(
        self, size_q1, size_q2, range_q1=(-3.0, 3.0), range_q2=(-3.0, 3.0), noise=0.0
    ):
        # sample q1 and q2 coordinates along each axes
        q1 = np.linspace(range_q1[0], range_q1[1], size_q1)
        q2 = np.linspace(range_q2[0], range_q2[1], size_q2)
        # combine q1 and q2 coordinates into a grid
        data = np.array(np.meshgrid(q1, q2)).T.reshape(-1, 2)
        return data


def select_monte_carlo(potential, min_potential, k, seed=42):
    npoints = len(potential)
    # compute Boltzman probability
    prob = np.exp(-1.5 * k * (potential - min_potential))

    rng = np.random.default_rng(seed=seed)
    random = rng.random(size=npoints)
    indices = np.where(prob > random)[0]
    print("Monte Carlo selection with k:", k)
    print("Minimum potential           :", min_potential)
    print("Total number of points      :", npoints)
    print("Number of selected points   :", len(indices))
    return indices


def generate_datasets(size_random, k_values, size_dataset, niter=10, seed=42):
    # Create directory for storing plots and npz data files, if it does not exist
    os.makedirs("data", exist_ok=True)

    # Make an instance of the 4-well potential
    # ----------------------------------------
    pes = FourWellPotential()
    print("Coordinates of the 4 minima :", pes.minima[0])
    print("Potential at the 4 minima   :", pes.minima[1])

    # Plot 2D potential using uniform grid
    # ------------------------------------
    points = pes.generate_points_uniform(size_q1=100, size_q2=100)
    values = pes.evaluate(points)
    print("Shape of PES points (for plotting):", points.shape)
    print("Shape of PES values (for plotting): ", values.shape)
    fname = "data/pes_4well.png"
    plot_data_2d(points, values, highlight=pes.minima[0], title="Four Well Potential", fname=fname)
    # 1D plot to see minimum D
    # ------------------------
    # q1_fixed = -1.17
    # points_1d = pes.generate_points(size_q1=1, size_q2=50, range_q1=(q1_fixed, q1_fixed))
    # values_1d = pes.evaluate(points_1d)
    # plt.plot(points_1d[:, 1], values_1d)
    # plt.xlabel("q2", fontsize=14)
    # plt.ylabel("Potential (kcal/mol)", fontsize=14)
    # plt.title(f"Intersection of PES at q1={q1_fixed}", fontsize=18)
    # plt.show()

    # Generate Dataset
    # ----------------
    # Select a set of random 2D points between -3.0 and 3.0. For large sizes (e.g., 10M),
    # this gives a dense and uniform coverage of the 4-well potential.
    # rng.random generates points in [0, 1] interval, so they are shifted ans scaled
    # to cover -3.0 to 3.0 range
    rng = np.random.default_rng(seed=seed)
    points_random = (rng.random((int(size_random), 2)) - 0.5) * 6.0
    values_random = pes.evaluate(points_random)
    # plot_data_2d(points, values, highlight=points_random, title="Four Well Potential (kcal/mol)")

    # Use Monte-Carlo (MC) method to select points for a given k, which results in a different
    # number of points for each k. The idea is that as k increases, the dataset becomes
    # more biased so we can better demonstrate the importance of diverse subset sampling.
    # Note that k=0 gives back the entire set, because probability of all points is one.
    # However, as k increases the number of points selected by MC becomes smaller. This causes
    # an unfair comparison, when we select a percentage of the data (for each k) as a subset.
    # E.g., 1% of 10M (k=0) is already going to be diverse, no matter how the selection is
    # made while 1% of 10K (k=30) is not going to be diverse in general.
    # So, we select a fixed number of data generated by MC (for all k values) for the rest
    # of our analysis. Keep in mind that this selection will be random so that it has a
    # similar distribution (i.e, the same statistical bias) as the original MC sample set.
    # Knowing that k=20 for 10M points, selected ~15K points, we choose 10K for fixed size.
    for k in k_values:
        indices_mc = select_monte_carlo(values_random, np.min(pes.minima[1]), k=k, seed=seed)
        # points_mc = points_random[indices_mc]
        dataset = {}
        # make subdirectory for each k to store data
        folder = f"data/monte_carlo_k{k:02d}"
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f"{folder}/plots", exist_ok=True)
        # randomly select a fixed-size dataset from each k niter times
        for index in range(niter):
            indices_iter = np.random.choice(indices_mc, size_dataset, replace=False)
            print(f" Select {len(indices_iter)} points iteration {index}")
            # plot MC selected points and the fixed-size dataset
            fname = f"dataset_{size_random:2.1e}_k{k:02d}_size{size_dataset:05d}_iter{index:02d}"
            plot_data_2d(
                points,
                values,
                highlight=(points_random[indices_mc], points_random[indices_iter]),
                title=f"Monte Carlo (k={k}): {len(indices_iter)} out of {len(indices_mc)}",
                fname=f"{folder}/plots/{fname}.png",
            )
            # store data for given k and iteration
            # dataset[fname] = points_random[indices_iter]
            dataset[fname] = {
                "points": points_random[indices_iter].tolist(),
                "values": values_random[indices_iter].tolist(),
            }
        # save data for given k
        # np.savez_compressed(f"data/k{k:02.1f}/{fname.split('_iter')[0]}", **dataset)
        print(f"Write {niter} iterations JSON data into data/{folder}")
        with open(f"{folder}/{fname.split('_iter')[0]}.json", "w") as fp:
            json.dump(dataset, fp, indent=4, sort_keys=True)


def select_subsets(points, percentages, methods, seed=42):
    rng = np.random.default_rng(seed=seed)
    selected_indices = {}
    # loop over methods first to avoid recomputing points_dist for MaxMin and MaxSum
    for method in methods:
        print(f" Selection Method: {method}")
        if method not in selected_indices:
            selected_indices[method] = {}
        points_dist = None
        if method in ["MaxMin", "MaxSum"]:
            points_dist = pairwise_distances(points, metric="euclidean")
        for p in percentages:
            # make an entry for each percentage
            if p not in selected_indices[method]:
                # str(p) is used as key because JSON does not allow storing int keys
                selected_indices[method][str(p)] = {}
            size = int(p * len(points) / 100)
            print(f"  Percentage={p: 2d}: Select {size} from {len(points)}")
            if method == "Random":
                indices = rng.choice(np.arange(len(points)), size, replace=False, shuffle=True)
            elif method == "MaxMin":
                indices = MaxMin(fun_dist=None).select(points_dist, size=size)
            elif method == "MaxSum":
                indices = MaxSum().select(points_dist, size=size)
            elif method == "DISE":
                indices = DISE().select(points, size=size)
            elif method == "OptiSim":
                indices = OptiSim().select(points, size=size)
            elif method == "Medoid":
                indices = Medoid().select(points, size=size)
            elif method.startswith("GridPartition"):
                grid_method = method.split("-")[1]
                indices = GridPartition(nbins_axis=5, bin_method=grid_method).select(
                    points, size=size
                )
            else:
                raise ValueError(f"Unknown selection method: {method}")
            # str(p) is used as key because JSON does not allow storing int keys
            # indices are converted to int because JSON does not allow storing numpy.int64
            selected_indices[method][str(p)] = [int(item) for item in indices]
    return selected_indices


def compute_diversity(points, selected_indices, measures):
    diversity = {}
    for measure in measures:
        if measure == "LogDet":
            diversity[measure] = logdet(points[selected_indices])
        elif measure == "WDUD":
            diversity[measure] = wdud(points[selected_indices])
        elif measure == "MinDist":
            points_dist = pairwise_distances(points[selected_indices], metric="euclidean")
            upper_triangular = points_dist[np.triu_indices_from(points_dist, k=1)]
            diversity[measure] = np.min(upper_triangular)
        elif measure == "MeanDist":
            points_dist = pairwise_distances(points[selected_indices], metric="euclidean")
            upper_triangular = points_dist[np.triu_indices_from(points_dist, k=1)]
            diversity[measure] = np.mean(upper_triangular)
        else:
            raise ValueError(f"Unknown diversity measure: {measure}")
    return diversity


def plot_data_2d(points, values, highlight=None, title="", fname=None):
    """Plot 2D data with colorbar.

    Parameters
    ----------
    points : np.ndarray
        2D array with 2 columns representing coordinates.
    values : np.ndarray
        1D array with potential values for each point used for coloring.
    highlight : np.ndarray
        2D array with 2 columns representing coordinates of points to highlight
        with a star marker.
    title : str
        Title of the plot.
    """
    if points.ndim != 2 and points.shape[1] != 2:
        raise ValueError("Argument points must be 2D array with 2 columns")
    plt.scatter(points[:, 0], points[:, 1], c=values, cmap="RdBu")
    plt.colorbar(label="Potential (kcal/mol)")
    if highlight is not None:
        if isinstance(highlight, np.ndarray) and highlight.ndim == 2 and highlight.shape[1] == 2:
            # raise ValueError("Argument highlight must be 2D array with 2 columns")
            plt.scatter(highlight[:, 0], highlight[:, 1], c="black", marker="*", s=20)  # , s=100)
        else:
            for index, item in enumerate(highlight):
                assert item.ndim == 2 and item.shape[1] == 2
                if index == 0:
                    plt.scatter(item[:, 0], item[:, 1], c="black", marker="*", s=50)
                else:
                    plt.scatter(
                        item[:, 0],
                        item[:, 1],
                        marker="o",
                        facecolors="none",
                        edgecolors="cyan",
                        linewidths=2,
                        s=10,
                    )

    plt.title(title, fontsize=14)
    plt.xlabel("q1", fontsize=14)
    plt.ylabel("q2", fontsize=14)

    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # get command line arguments
    args = sys.argv[1:]

    if args[0] == "generate_data":
        # These parameter choices are discussed in generate_datasets function
        size_random, size_dataset = 1.0e7, 10000
        # k_values = np.arange(0.0, 21.0, 1.0, dtype=int)
        k_values = [2, 10]
        generate_datasets(size_random, k_values, size_dataset, niter=10, seed=42)

    elif args[0] == "select_subsets":
        fnames = sorted(args[1:])
        # list of selector methods and percentages to select
        selectors = [
            "Random",
            "MaxMin",
            "MaxSum",
            "OptiSim",
            "DISE",
            "Medoid",
            "GridPartition-equisized_dependent",
            "GridPartition-equisized_independent",
            "GridPartition-equifrequent_dependent",
            "GridPartition-equifrequent_independent",
        ]
        percentages = [2**i for i in range(7)]

        # Make an instance of potential for visualization
        # -----------------------------------------------
        pes = FourWellPotential()
        points_plot = pes.generate_points_uniform(size_q1=100, size_q2=100)
        values_plot = pes.evaluate(points_plot)

        # Loop over datasets, select subsets, plot and store selected indices
        # -------------------------------------------------------------------
        # example of fname: data/monte_carlo_k00/dataset_1.0e+06_k00_size01000.json
        for fname in fnames:
            # Load dataset from JSON file
            # ---------------------------
            print(f"LOAD {fname}")
            with open(fname, "r") as fp:
                data = json.load(fp)
            folder, fname = os.path.split(fname)
            os.makedirs(f"{folder}/plots", exist_ok=True)
            # Loop over iterations of dataset
            # -------------------------------
            database_indices = {}
            for fname_iter, values_iter in data.items():
                print(f" SELECT from: {fname_iter}")
                points_iter = np.array(values_iter["points"])
                # values_iter = np.array(values_iter["values"])
                # example of fname_points: dataset_1.0e+07_k10.0_size10000_iter00
                # Select subsets
                # --------------
                # indices are returned as a nested dictionary {method: {percentage: indices}}
                selected_indices = select_subsets(points_iter, percentages, selectors, seed=42)
                database_indices[fname_iter] = selected_indices
                print("DONE SELECTING")
                # Plot Subsets
                # ------------
                for method, values in selected_indices.items():
                    print("PLOTTING.....")
                    os.makedirs(f"{folder}/plots/{method}", exist_ok=True)
                    for p, indices in values.items():
                        p = int(p)
                        fname_plot = (
                            f"{folder}/plots/{method}/subset_{method}_p{p:02d}_{fname_iter}.png"
                        )
                        print(f"PLOT {fname_plot}")
                        plot_data_2d(
                            points_plot,
                            values_plot,
                            highlight=(points_iter, points_iter[indices]),
                            title=f"{fname_iter}\nMethod={method}, Percentage={p}, Size={len(indices)}",
                            fname=fname_plot,
                        )
            # Save selected indices from fname database
            # -----------------------------------------
            for method in selectors:
                temp = {fname: values[method] for fname, values in database_indices.items()}
                fname_json = fname.replace("dataset", f"subset_{method}_dataset")
                fname_json = f"{folder}/{fname_json}"
                print(f"WRITE {fname_json}")
                with open(fname_json, "w") as fp:
                    json.dump(temp, fp, indent=4, sort_keys=True)

    elif args[0] == "compute_diversity":
        fnames = sorted(args[1:])
        if not fnames:
            raise ValueError("No dataset JSON files were provided.")

        selectors = [
            "Random",
            "MaxMin",
            "MaxSum",
            "OptiSim",
            "DISE",
            "Medoid",
            "GridPartition-equisized_dependent",
            "GridPartition-equisized_independent",
            "GridPartition-equifrequent_dependent",
            "GridPartition-equifrequent_independent",
        ]
        measures = ["LogDet", "WDUD"]

        diversity_results = {}

        for dataset_path in fnames:
            print(f"LOAD DATASET {dataset_path}")

            # break
            with open(dataset_path, "r") as fp:
                dataset_entries = json.load(fp)

            folder, dataset_fname = os.path.split(dataset_path)
            dataset_key = dataset_path.replace("data/", "", 1)
            diversity_dataset = diversity_results.setdefault(dataset_key, {})
            # diversity_dataset = {}

            # Load subset index files for all selector methods
            subset_indices = {}
            for method in selectors:
                subset_fname = dataset_fname.replace("dataset", f"subset_{method}_dataset")
                subset_path = os.path.join(folder, subset_fname)
                if not os.path.exists(subset_path):
                    print(f" SKIP {method}: missing subset file {subset_path}")
                    continue
                print(f"LOAD SUBSET {subset_path}")
                with open(subset_path, "r") as fp:
                    subset_indices[method] = json.load(fp)
                diversity_dataset.setdefault(method, {})

            # Iterate over dataset iterations and compute diversity for each subset
            for iteration_name, iteration_values in dataset_entries.items():
                points_iter = np.array(iteration_values["points"], dtype=float)
                for method, method_indices in subset_indices.items():
                    if iteration_name not in method_indices:
                        print(
                            f" SKIP method={method}: iteration {iteration_name} not found in subset file."
                        )
                        continue
                    for percentage, indices in method_indices[iteration_name].items():
                        if not indices:
                            print(
                                f" SKIP method={method}, iteration={iteration_name}, percentage={percentage}: empty selection."
                            )
                            continue
                        indices_array = np.asarray(indices, dtype=int)
                        diversity_values = compute_diversity(points_iter, indices_array, measures)

                        method_store = diversity_dataset.setdefault(method, {})
                        percentage_store = method_store.setdefault(percentage, {})
                        for measure, value in diversity_values.items():
                            percentage_store.setdefault(measure, []).append(float(value))

            # Save per-dataset diversity summary next to dataset file
            diversity_out_fname = dataset_fname.replace("dataset", "diversity")
            diversity_out_path = os.path.join(folder, diversity_out_fname)
            print(f"WRITE {diversity_out_path}")
            with open(diversity_out_path, "w", encoding="utf-8") as fp:
                json.dump(diversity_dataset, fp, indent=4, sort_keys=True)

        # Save combined diversity results
        os.makedirs("data", exist_ok=True)
        # dataset_path
        # data/monte_carlo_k02/dataset_1.0e+07_k02_size10000.json
        diversity_master_path = dataset_path.replace("dataset", "diversity")
        print(f"WRITE {diversity_master_path}")
        with open(diversity_master_path, "w", encoding="utf-8") as fp:
            json.dump(diversity_results, fp, indent=4, sort_keys=True)

        # Plot summary metrics for quick inspection
        plot_measures = ["LogDet", "WDUD"]
        for measure in plot_measures:
            for dataset_key, dataset_values in diversity_results.items():
                if not dataset_values:
                    continue
                plt.figure()
                for method, method_values in dataset_values.items():
                    if not method_values:
                        continue
                    percentages = sorted(int(p) for p in method_values.keys())
                    averages = []
                    for percentage in percentages:
                        measure_values = method_values[str(percentage)].get(measure, [])
                        if not measure_values:
                            averages.append(np.nan)
                        else:
                            averages.append(float(np.mean(measure_values)))
                    plt.plot(percentages, averages, marker="o", label=method)
                plt.xlabel("Percentage (%)")
                plt.ylabel(measure)
                plt.title(dataset_key)
                plt.legend()
                plot_fname = dataset_key.replace("/", "_").replace(".json", f"_{measure}.png")
                # dataset_path
                plot_path = os.path.join(folder, "plots", plot_fname)
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                print(f"SAVE PLOT {plot_path}")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
    else:
        raise ValueError(f"Unknown task={args}")
