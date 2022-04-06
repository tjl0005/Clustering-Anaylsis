from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np


def scores(data, labels):
    """Get scores for the clusters produced"""
    # A measure of similarity to other clusters
    sc = np.round(metrics.silhouette_score(data, labels), 2)  # 1 is best, 0 worst
    # Measure of separation
    db = np.round(metrics.davies_bouldin_score(data, labels), 2)  # Lower is better, 0-1

    return sc, db


def vis_clusters(c_type, side, method, diff_df, clusters):
    """Produce scatter plot for the clusters"""
    score = scores(diff_df, clusters.labels_)
    score = ("Silhouette: {:.2f} \nDavies-Bouldin: {}\n".format(score[0], score[1]))
    diff_df.plot.scatter(x="Component 1", y="Component 2", c=clusters.labels_, cmap="tab20", label=score)

    # Plot centroids
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    # Display scores in best position
    plt.legend(handlelength=0, frameon=False, handletextpad=0)

    title = "{} using {} Scaling with {}".format(c_type, method[0], method[1])

    plt.title(title)
    plt.savefig("../Visualisations/Clusters/{}/{}/{}.png".format(side, c_type, title), dpi=600)


def vis_results(score, vals, param, c_params, c_type, method, side):
    """Produce subplot of scores for a parameter sweep"""
    fig, ax = plt.subplots(2, 2, tight_layout=True)

    if isinstance(vals[0], int):
        ax[0, 0].grid(alpha=0.5)
        ax[0, 0].plot(vals, score[0], 'o--', c="tab:purple")

        ax[0, 1].grid(alpha=0.5)
        ax[0, 1].plot(vals, score[1], 'o--', c="tab:green")

    else:
        ax[0, 0].grid(alpha=0.5, axis='y')
        ax[0, 0].bar(vals, score[0], width=0.5, color="tab:purple")

        ax[0, 1].grid(alpha=0.5, axis='y')
        ax[0, 1].bar(vals, score[1], width=0.5, color="tab:green")

    # Silhouette Score
    ax[0, 0].set_title("Silhouette Coefficient")
    ax[0, 0].set_ylabel("Score")
    ax[0, 0].set_xlabel(param)

    # Davies-Bouldin Score
    ax[0, 1].set_title("Davies-Bouldin")
    ax[0, 1].set_ylabel("Score")
    ax[0, 1].set_xlabel(param)

    p_one = "{}: {}".format(c_params[0][0], c_params[0][1])
    p_two = "{}: {}".format(c_params[1][0], c_params[1][1])

    # Use last subplot to show the current setup
    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title("{}".format(c_type))
    ax[1, 1].text(0.1, 0.9, "Parameters:", horizontalalignment="center")
    ax[1, 1].text(0.3, 0.7, "{}\n{}".format(p_one, p_two), horizontalalignment="center", fontsize=8)
    ax[1, 1].text(0.1, 0.5, "Description:", horizontalalignment="center")
    ax[1, 1].text(0.5, 0.4, "For {} Side using {} Scaling with {}".format(side, method[0], method[1]),
                  horizontalalignment="center", fontsize=8, wrap=True)

    plt.savefig("../Visualisations/Parameter Sweeps/{}/{}/{}.png".format(side, c_type, param), dpi=600)
