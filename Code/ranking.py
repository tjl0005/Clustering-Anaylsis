from Code.clustering import kmeans, hierarchical, density, scores
from matplotlib import pyplot as plt
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
        davies.append(score[1])
        calinski.append(score[2])

    metrics = ["Silhouette Coefficient", "Davies-Bouldin", "Calinski-Harabasz"]
    C_scores = [silhouette, davies, calinski]

    plot_results(C_scores, vals, metrics, param)

    # Get rankings for each score
    print("Rankings:")
    for i in range(3):
        rank_results(C_scores[i], vals, metrics[i])


def rank_results(score, vals, metric):
    """Return list of indexes for sorted list"""
    # Sort and return IDs
    s_scores = np.argsort(score)

    print("{}:".format(metric))

    if metric == "Calinski":  # Low to high
        for i in s_scores:
            print("Score: {} from value: {}".format(score[i], vals[i]))
    else:
        for i in reversed(s_scores):  # High to low
            print("Score: {} from value: {}".format(score[i], vals[i]))

    print("\n")


def plot_results(score, vals, metrics, param):
    """Produce subplot of scores for a parameter sweep"""
    fig, ax = plt.subplots(2, 2, tight_layout=True)

    # Silhouette Score
    ax[0, 0].grid(alpha=0.5)
    ax[0, 0].plot(vals, score[0], label="Aiming for 1")
    ax[0, 0].set_title(metrics[0])
    ax[0, 0].set_ylabel("Score")
    ax[0, 0].set_xlabel(param)

    # Davies-Bouldin Score
    ax[0, 1].grid(alpha=0.5)
    ax[0, 1].plot(vals, score[1], label="Aiming for 0")
    ax[0, 1].set_title(metrics[1])
    ax[0, 1].set_ylabel("Score")
    ax[0, 1].set_xlabel(param)

    # Calinski-Harabasz Score
    ax[1, 0].grid(alpha=0.5)
    ax[1, 0].plot(vals, score[2], label="Higher is better")
    ax[1, 0].set_title(metrics[2])
    ax[1, 0].set_ylabel("Score")
    ax[1, 0].set_xlabel(param)

    ax[1, 1].set_axis_off()

    # Score descriptions
    ax[0, 0].legend(handlelength=0, handletextpad=0)
    ax[0, 1].legend(handlelength=0, handletextpad=0)
    ax[1, 0].legend(handlelength=0, handletextpad=0)

    plt.show()
