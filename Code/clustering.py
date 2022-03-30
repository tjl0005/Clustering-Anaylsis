from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np


def scores(data, labels, display):
    """Get scores for the clusters produced"""
    # A measure of similarity to other clusters
    silhouette = np.round(metrics.silhouette_score(data, labels), 2)  # 1 is best, 0 worst
    # Measure of separation
    davies_bouldin = np.round(metrics.davies_bouldin_score(data, labels), 2)  # Lower is better, 0-1
    # Dispersion of clusters, is the current number of clusters good
    calinski_harabasz = np.round(metrics.calinski_harabasz_score(data, labels), 2)  # Higher is better

    # Decide on output method
    if display:
        return ("Silhouette: {}\nDavies-Bouldin: {}\nCalinski-Harabasz: {}\n"
                .format(silhouette, davies_bouldin, calinski_harabasz))
    else:
        return silhouette, davies_bouldin, calinski_harabasz


def scatter_plot(df, score, clusters, c_type):
    """Produce scatter plot for the clusters"""
    labels = df.columns.values

    # Scatter plot with all labelled values representing clusters
    df.plot.scatter(x=labels[0], y=labels[1], c=clusters.labels_, cmap="rainbow", label=score)

    # Plot centroids
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    # Display scores in best position
    plt.legend(handlelength=0, handletextpad=0)

    plt.title(c_type)
    plt.show()


def kmeans(data, k_params, display):
    """Produce clusters using K-Means"""
    clustering = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    clustering.fit(data)

    # Decide how to display results
    if display == "plot":
        scatter_plot(data, scores(data, clustering.labels_, True), clustering, "K-Means")
    elif display == "print":
        print(scores(data, clustering.labels_, True))
    else:
        return data, clustering.labels_


def hierarchical(data, h_params, display):
    """Produce clusters using Agglomerative Clustering"""
    clustering = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    clustering.fit(data)

    if display == "plot":
        scatter_plot(data, scores(data, clustering.labels_, True), clustering, "Hierarchical")
    elif display == "print":
        print(scores(data, clustering.labels_, True))
    else:
        return data, clustering.labels_


def density(data, d_params, c_type, display):
    """Produce clusters using either DBSCAN or OPTICS"""
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        db_c = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        db_c = OPTICS(min_samples=d_params["min_samples"], p=d_params["p"],
                      min_cluster_size=d_params["min_cluster_size"])

    db_c.fit(data)

    if display == "plot":
        scatter_plot(data, scores(data, db_c.labels_, True), db_c, c_type)
    elif display == "print":
        print(scores(data, db_c.labels_, True))
    else:
        return scores(data, db_c.labels_, False)
