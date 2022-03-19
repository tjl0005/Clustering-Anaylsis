from Code.clustering import kmeans_clustering, hierarchical_clustering, density_clustering, get_scores
import numpy as np


def param_sweep(c_type, c_params, data, param, vals):
    silhouette = []
    davies = []
    calinski = []

    # Go through all potential parameter values
    for val in vals:
        # Update specified parameter value
        c_params[param] = val

        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            data, labels = kmeans_clustering(data, c_params, "score")
        elif c_type == "Hierarchical":
            data, labels = hierarchical_clustering(data, c_params, "score")
        else:
            data, labels = density_clustering(data, c_params, c_type, "score")

        scores = get_scores(data, labels, c_type, "")

        silhouette.append(scores[0])
        calinski.append(scores[1])
        davies.append(scores[2])

    rank_results(silhouette, vals, "Silhouette")
    rank_results(calinski, vals, "Calinski")
    rank_results(davies, vals, "Davies")


def rank_results(scores, vals, metric):
    # Return list of indexes for sorted list
    s_scores = np.argsort(scores)

    print("\n{}:".format(metric))

    if metric == "Calinski":  # Low to high
        for i in s_scores:
            print("Score: {} from value: {}".format(scores[i], vals[i]))
    else:
        for i in reversed(s_scores):  # High to low
            print("Score: {} from value: {}".format(scores[i], vals[i]))
