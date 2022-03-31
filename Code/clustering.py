from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np


def run_ca(c_type, df, params):
    """Produce and display clustering algorithms using specified method"""
    if c_type == "K-Means":
        kmeans(df, params["kmeans"], "plot")
    elif c_type == "Hierarchical":
        hierarchical(df, params["hierarchical"], "plot")
    elif c_type == "DBSCAN":
        density(df, params["dbscan"], "DBSCAN", "plot")
    else:
        density(df, params["optics"], "OPTICS", "plot")


def scores(data, labels):
    """Get scores for the clusters produced"""
    # A measure of similarity to other clusters
    silhouette = np.round(metrics.silhouette_score(data, labels), 2)  # 1 is best, 0 worst
    # Measure of separation
    davies_bouldin = np.round(metrics.davies_bouldin_score(data, labels), 2)  # Lower is better, 0-1
    # Dispersion of clusters, is the current number of clusters good
    calinski_harabasz = np.round(metrics.calinski_harabasz_score(data, labels), 2)  # Higher is better

    return silhouette, davies_bouldin, calinski_harabasz


def scatter_plot(c_type, diff_df, score, clusters):
    """Produce scatter plot for the clusters"""
    score = ("Silhouette: {}\nDavies-Bouldin: {}\nCalinski-Harabasz: {}\n".format(score[0], score[1], score[2]))

    diff_df.plot.scatter(x="Principal Component 1", y="Principal Component 2", c=clusters.labels_, cmap="tab20", label=score)

    # Plot centroids
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    # Display scores in best position
    plt.legend(handlelength=0, frameon=False, handletextpad=0)

    plt.title(c_type)
    plt.savefig("../Visualisations/Clusters/{}.png".format(c_type), dpi=600)


def kmeans(data, k_params, display):
    """Produce clusters using K-Means"""
    clustering = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    clustering.fit(data)

    # Decide how to display results
    if display == "plot":
        scatter_plot("K-Means", data, scores(data, clustering.labels_), clustering)
    else:
        return data, clustering.labels_


def hierarchical(data, h_params, display):
    """Produce clusters using Agglomerative Clustering"""
    clustering = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    clustering.fit(data)

    if display == "plot":
        scatter_plot("Hierarchical", data, scores(data, clustering.labels_), clustering)
    else:
        return data, clustering.labels_


def density(data, d_params, c_type, display):
    """Produce clusters using either DBSCAN or OPTICS"""
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        clustering = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        clustering = OPTICS(min_samples=d_params["min_samples"], cluster_method=d_params["cluster_method"])

    clustering.fit(data)

    if display == "plot":
        scatter_plot(c_type, data, scores(data, clustering.labels_), clustering)
    else:
        return data, clustering.labels_
