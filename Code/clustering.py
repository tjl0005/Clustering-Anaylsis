from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np


def run_ca(c_type, data, params, method):
    """Produce and display clustering algorithms using specified method"""
    if c_type == "K-Means":
        clusters = kmeans(data, params["kmeans"])
    elif c_type == "Hierarchical":
        clusters = hierarchical(data, params["hierarchical"])
    else:
        clusters = density(data, params[c_type], c_type)

    scatter_plot(c_type, method, data, clusters)


def scores(data, labels):
    """Get scores for the clusters produced"""
    # A measure of similarity to other clusters
    sc = np.round(metrics.silhouette_score(data, labels), 2)  # 1 is best, 0 worst
    # Measure of separation
    db = np.round(metrics.davies_bouldin_score(data, labels), 2)  # Lower is better, 0-1
    # Dispersion of clusters, is the current number of clusters good
    ch = np.round(metrics.calinski_harabasz_score(data, labels), 2)  # Higher is better

    return sc, db, ch


def scatter_plot(c_type, method, diff_df, clusters):
    """Produce scatter plot for the clusters"""
    score = scores(diff_df, clusters.labels_)
    score = ("SC: {:.2f} \nDB: {}\nCH: {}\n".format(score[0], score[1], score[2]))
    diff_df.plot.scatter(x="Component 1", y="Component 2", c=clusters.labels_, cmap="tab20", label=score)

    # Plot centroids
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    # Display scores in best position
    plt.legend(handlelength=0, frameon=False, handletextpad=0)

    title = "{} with {} Scaling and {}".format(c_type, method[0], method[1])

    plt.title(title)
    plt.savefig("../Visualisations/Clusters/{}.png".format(title), dpi=600)


def kmeans(data, k_params):
    """Produce clusters using K-Means"""
    clustering = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    clustering.fit(data)

    return clustering


def hierarchical(data, h_params):
    """Produce clusters using Agglomerative Clustering"""
    clustering = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    clustering.fit(data)

    return clustering


def density(data, d_params, c_type):
    """Produce clusters using either DBSCAN or OPTICS"""
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        clustering = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        clustering = OPTICS(min_samples=d_params["min_samples"], cluster_method=d_params["cluster_method"])

    clustering.fit(data)

    return clustering
