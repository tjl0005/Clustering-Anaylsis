from Code.clustering import kmeans, hierarchical, density, scores
import numpy as np


def param_sweep(c_type, c_params, data, param, vals):
    """Test algorithm with different parameter values"""
    silhouette = []
    davies = []
    calinski = []

    for val in vals:
        # Update specified parameter value
        c_params[param] = val

        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            data, labels = kmeans(data, c_params, "score")
        elif c_type == "Hierarchical":
            data, labels = hierarchical(data, c_params, "score")
        else:
            data, labels = density(data, c_params, c_type, "score")

        score = scores(data, labels, "")

        silhouette.append(score[0])
        calinski.append(score[1])
        davies.append(score[2])

    rank_results(silhouette, vals, "Silhouette")
    rank_results(calinski, vals, "Calinski")
    rank_results(davies, vals, "Davies")


def rank_results(score, vals, metric):
    """Return list of indexes for sorted list"""
    # Sort and return IDs
    s_scores = np.argsort(score)

    print("\n{}:".format(metric))

    if metric == "Calinski":  # Low to high
        for i in s_scores:
            print("Score: {} from value: {}".format(score[i], vals[i]))
    else:
        for i in reversed(s_scores):  # High to low
            print("Score: {} from value: {}".format(score[i], vals[i]))
