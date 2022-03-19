from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np


def get_scores(data, labels, c_type, display):
    # A measure of how dense and seperated the clusters are
    silhouette = np.round(metrics.silhouette_score(data, labels), 3)  # 1 is best, 0 worst
    # Measure of separation
    davies_bouldin = np.round(metrics.davies_bouldin_score(data, labels), 3)  # Lower is better
    # How effective the number of clusters are
    calinski_harabasz = np.round(metrics.calinski_harabasz_score(data, labels), 3)  # Higher is better

    if display:
        return ("Scores for {}:\n Silhouette Coefficient: {}\n Calinski Harabasz: {}\n Davies-Bouldin Index: {}\n"
                .format(c_type, silhouette, calinski_harabasz, davies_bouldin))
    else:
        return silhouette, calinski_harabasz, davies_bouldin


def plot_clusters(df, scores, clusters, c_type):
    # Remove ID so only relevant columns used
    df.drop("ID", axis=1, inplace=True)
    categories = df.columns.values

    # Scatter plot with all labelled values representing clusters
    df.plot.scatter(x=categories[0], y=categories[1], c=clusters.labels_, cmap="rainbow")

    # Plot center points of clusters
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    plt.title("{} Clusters for {}".format(c_type, categories))
    plt.text(8, 18, scores)  # Show scores in visualisation
    plt.show()


def kmeans_clustering(data, k_params, display):
    k_means = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    k_means.fit(data)

    # Decide how to display results
    if display == "plot":
        plot_clusters(data, get_scores(data, k_means.labels_, "K-Means", True), k_means, "K-Means")
    elif display == "print":
        print(get_scores(data, k_means.labels_, "K-Means", True))
    else:
        return data, k_means.labels_


def hierarchical_clustering(data, h_params, display):
    hierarchical = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    hierarchical.fit(data)

    if display == "plot":
        plot_clusters(data, get_scores(data, hierarchical.labels_, "Hierarchical", True), hierarchical,
                      "Hierarchical")
    elif display == "print":
        print(get_scores(data, hierarchical.labels_, "Hierarchical", True))
    else:
        return data, hierarchical.labels_


def density_clustering(data, d_params, c_type, display):
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        db_c = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        db_c = OPTICS(min_samples=d_params["min_samples"], p=d_params["p"],
                      min_cluster_size=d_params["min_cluster_size"])

    db_c.fit(data)

    if display == "plot":
        plot_clusters(data, get_scores(data, db_c.labels_, c_type, True), db_c, c_type)
    elif display == "print":
        print(get_scores(data, db_c.labels_, c_type, True))
    else:
        return get_scores(data, db_c.labels_, c_type, False)
