from clustering import run_ca
from data import proc_data
from visualise import vis_results, scores
from prettytable import PrettyTable
import numpy as np


def sweep(c_type, side, params, method):
    """"Perform all parameter sweeps for a specified algorithm"""
    data = proc_data(side[1], method)
    print("Results for: {}".format(c_type))

    if c_type == "K-Means":
        param_sweep(c_type, params[c_type], data, method, side[0], "n_clusters", range(4, 21))
    elif c_type == "Hierarchical":
        param_sweep(c_type, params[c_type], data, method, side[0], "linkage", ["ward", "complete"])
    elif c_type == "DBSCAN":
        # param_sweep(c_type, params[c_type], data, method, side[0], "min_samples", range(0, 10))
        param_sweep(c_type, params[c_type], data, method, side[0], "eps", np.arange(0.1, 2.1, 0.1))
    else:
        if params[c_type]["preference"] != "None":
            param_sweep(c_type, params[c_type], data, method, side[0], "preference", np.arange(-20, -1, 1))
        else:
            param_sweep(c_type, params[c_type], data, method, side[0], "damping", np.arange(0.5, 1.01, 0.1))


def param_sweep(c_type, c_params, data, method, side, param, vals):
    """Test algorithm with different parameter values"""
    silhouette = []
    davies = []

    if not isinstance(vals[0], str):
        for val in vals:
            # Update specified parameter value
            c_params[param] = val

            # Store scores for use
            score = scores(data, run_ca(c_type, data, c_params).labels_)
            silhouette.append(score[0])
            davies.append(score[1])

        c_scores = [silhouette, davies]
        print_results(c_scores, vals, param)

    # Sweeping linkage around clusters
    else:
        for linkage in vals:  # Go through each linkage type
            # Store scores for this linkage type
            l_sc, l_db = [], []
            for n in range(4, 21):  # Always testing upto 15 clusters
                # Parameters to use for the sweep
                c_params["n_clusters"], c_params["linkage"] = n, linkage
                # Get scores for this linkage type
                s, d = scores(data, run_ca("Hierarchical", data, c_params).labels_)
                l_sc.append(s), l_db.append(d)

            # Contains all scores for every linkage type
            silhouette.append(l_sc), davies.append(l_db)
        # Prepare to pass all scores
        c_scores = [silhouette, davies]

    # Display results
    vis_results(c_scores, vals, param, c_type, method, side)


def print_results(c_scores, vals, param):
    """Print table to console displaying cluster scores"""
    table = PrettyTable()
    table.add_column(param, vals)
    table.add_column("Silhouette", c_scores[0])
    table.add_column("Davies-Bouldin", c_scores[1])

    print(table)
