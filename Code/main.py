from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob


# Create dataframes for all data and drop irrelevant data
def prepare_data():
    # List of all Excel files
    files = glob.glob("../data/*.xls")
    dfs = []

    # Go through all xls files
    for i in range(len(files)):
        # Add dataframe to list
        dfs.append(pd.read_excel(files[i], sheet_name=0))

        # Remove irrelevant columns
        if "Error" in dfs[i].columns:
            dfs[i].drop("Error", axis=1, inplace=True)
        elif "Warning" in dfs[i].columns:
            dfs[i].drop("Warning", axis=1, inplace=True)

    # All dataframes
    return dfs


def get_scores(dfs, labels):
    silhouette = metrics.silhouette_score(dfs, labels)  # 1 is best, 0 worst
    calinski_harabasz = metrics.calinski_harabasz_score(dfs, labels)  # Higher is better
    davies_bouldin = metrics.davies_bouldin_score(dfs, labels)  # Lower is better, 0-1

    # Formatted string to show scores
    return ("Silhouette Coefficient: {}\n Calinski-Harabasz Index: {}\n Davies-Bouldin Index: {}"
            .format(np.round(silhouette, 3), np.round(calinski_harabasz, 3), np.round(davies_bouldin, 3)))


def plot_clusters(dfs, categories, scores, clusters, c_type):
    # Scatter plot with all labelled values representing clusters
    dfs.plot.scatter(x=categories[0], y=categories[1], c=clusters.labels_, cmap='rainbow')

    if c_type == "K-Means":
        # Center points of the clusters, represented by an X
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X")

    plt.title("{} Clustering".format(c_type))
    plt.text(8, 18, scores)
    plt.show()


def kmeans_clustering(dfs, categories, n, init, max_i):
    # Number of clusters, run times, max iterations for one run
    kmeans = KMeans(n_clusters=n, n_init=init, max_iter=max_i)
    kmeans.fit(dfs)
    c_scores = get_scores(dfs, kmeans.labels_)

    print("Scores for K-Means:\n " + c_scores)
    plot_clusters(dfs, categories, c_scores, kmeans, "K-Means")


def hierarchical_clustering(dfs, categories, n, linkage):
    # Number of clusters, linkage, linkage criterion
    hierarchical = AgglomerativeClustering(n_clusters=n, linkage=linkage)
    hierarchical.fit(dfs)
    c_scores = get_scores(dfs, hierarchical.labels_)

    print("\nScores for Hierarchical:\n " + c_scores)
    plot_clusters(dfs, categories, c_scores, hierarchical, "Hierarchical")


def density_clustering(dfs, categories, c_type):
    if c_type == "DBSCAN":
        # Max distances between samples, number of samples for core point
        db_c = DBSCAN(eps=0.5, min_samples=5)
        db_c.fit(dfs)
    else:
        # Number of samples for core point, distance type, min number of samples for a cluster
        db_c = OPTICS(min_samples=5, p=2, min_cluster_size=None)
        db_c.fit(dfs)

    c_scores = get_scores(dfs, db_c.labels_)

    plot_clusters(dfs, categories, c_scores, db_c, c_type)
    print("\nScores for {}:\n ".format(c_type) + c_scores)


all_data = prepare_data()

# Accessing the first dataframe, and 2 categories to use
test_categories = ["Emin Medial", "Emin Lateral"]
test_data = all_data[0][test_categories]

# # Run clustering algorithms, plot respective clusters and print scores
kmeans_clustering(test_data, test_categories, 10, 10, 300)
hierarchical_clustering(test_data, test_categories, 10, "ward")
density_clustering(test_data, test_categories, "DBSCAN")
density_clustering(test_data, test_categories, "OPTICS")
