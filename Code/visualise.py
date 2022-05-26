from matplotlib import pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np


def scores(data, labels):
    """Get scores for the clusters produced"""
    if len(set(labels)) == 1:
        return -1, -1
    else:
        # A measure of similarity to other clusters
        sc = np.round(metrics.silhouette_score(data, labels), 2)  # 1 is best, 0 worst
        # Measure of separation
        db = np.round(metrics.davies_bouldin_score(data, labels), 2)  # Lower is better, 0-1

        return sc, db


def vis_results(score, vals, param, c_type, method, side):
    """Produce subplot of scores for a parameter sweep"""
    fig, ax = plt.subplots(2, 1, tight_layout=True)

    if not isinstance(vals[0], str):
        # ax[0].set_ylim([-1, 0.36])
        # ax[1].set_ylim([0.5, 0.7])

        ax[0].grid(alpha=0.5)
        ax[0].plot(vals, score[0], 'o--', c="tab:purple")
        ax[0].set_xlabel(param)

        ax[1].grid(alpha=0.5)
        ax[1].plot(vals, score[1], 'o--', c="tab:green")
        ax[1].set_xlabel(param)

    # Linkage visualisation
    else:
        ax[0].set_ylim([0.25, 0.4])
        ax[1].set_ylim([0.6, 1.1])
        for i in range(0, 2):  # Two potential linkage types
            ax[0].plot(range(4, 21), score[0][i], 'o--', label=vals[i])
            ax[1].plot(range(4, 21), score[1][i], 'o--', label=vals[i])

        ax[0].grid(alpha=0.5, axis='y')
        ax[0].set_xlabel("n_clusters")
        ax[0].legend(bbox_to_anchor=(1, 1.2))

        ax[1].grid(alpha=0.5, axis='y')
        ax[1].set_xlabel("n_clusters")

    # Plot setup the same for all parameters at this point
    ax[0].set_title("Silhouette Coefficient")
    ax[0].set_ylabel("Score")
    ax[1].set_title("Davies-Bouldin")
    ax[1].set_ylabel("Score")

    title = "{} Scaling using {}".format(method["scaling"], method["reduction"])
    fig.suptitle(title)
    plt.savefig("../Visualisations/{} Indexes/Parameter Sweeps/{}/{} {}.png".format(side, c_type, param, title),
                bbox_inches='tight', dpi=600)
    plt.show()


def vis_compare(side, data, clusters, methods, alg):
    """Produce visualisations for provided clusters"""
    # Decide figure details
    if len(clusters) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axe = axes.flatten()
    default = methods  # Used to reset
    i = 0
    eps = [0.1, 0.3, 1.5, 2]

    for ax in axe:
        centroid = True  # Decide if centroids need to be plotted
        methods = default  # Reset

        if len(methods) == 4:  # If there are multiple methods prepare for them to be iterable
            method = list(methods)[i]
            methods = methods[method]

        if isinstance(alg, list):  # If multiple algorithm types being tested
            if isinstance(alg[i], str):  # Update title
                title = "{} using {} Scaling with {}".format(alg[i], methods["scaling"],
                                                             methods["reduction"])
                if alg[i] != "Affinity Propagation" or alg[i] != "K-Means":  # Don't plot centroids for hierarchical
                    centroid = False
            else:
                title = "Preference -{}".format(alg[i])
        else:  # Single method
            title = "{} using {} Scaling with {}".format(alg, methods["scaling"], methods["reduction"])
            if alg != "Affinity Propagation" or alg[i] != "K-Means":
                centroid = False

        title = "DBSCAN with eps {}".format(eps[i])

        df = pd.DataFrame(data[i])
        c_score = scores(data[i], clusters[i].labels_)
        score = ("Silhouette: {:.2f} \nDavies-Bouldin: {}\n".format(c_score[0], c_score[1]))

        df.plot.scatter(ax=ax, x="Component 1", y="Component 2", c=clusters[i].labels_, cmap="tab20", label=score)
        ax.legend(handlelength=0, frameon=False, handletextpad=0)

        ax.set_title("Preference -1 and Damping 0.6")
        if centroid:
            ax.scatter(clusters[i].cluster_centers_[:, 0], clusters[i].cluster_centers_[:, 1], marker="X", c="black")

        i += 1
    # plt.savefig("../Visualisations/{} Indexes/Clusters/{}/cluster_comparison.png".format(side, alg),
    #             bbox_inches='tight', dpi=600)
    # plt.savefig("Another one.png")
    plt.show(dpi=1000)
