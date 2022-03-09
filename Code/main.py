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


def kmeans_clustering(dfs, columns, n, init, max_i):
    # Number of clusters, run times, max iterations for one run
    kmeans = KMeans(n_clusters=n, n_init=init, max_iter=max_i)
    kmeans.fit(dfs)
    c_scores = get_scores(dfs, kmeans.labels_)

    # Scatter plot with all labelled values representing clusters
    dfs.plot.scatter(x=columns[0], y=columns[1], c=kmeans.labels_, cmap='rainbow')
    # Center points of the clusters, represented by an X
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X")

    plt.title("K-Means Clustering with {} Clusters".format(n))
    plt.text(8, 18, c_scores)
    plt.show()

    print("Scores for K-Means:\n " + c_scores)


def hierarchical_clustering(dfs, columns, n, linkage):
    # Number of clusters, linkage, linkage criterion
    hierarchical = AgglomerativeClustering(n_clusters=n, linkage=linkage)
    hierarchical.fit(dfs)
    c_scores = get_scores(dfs, hierarchical.labels_)

    dfs.plot.scatter(x=columns[0], y=columns[1], c=hierarchical.labels_, cmap='rainbow')

    plt.title("Hierarchical Clustering with {} Clusters".format(n))
    plt.text(8, 18, c_scores)
    plt.show()

    print("\nScores for Hierarchical:\n " + c_scores)


def density_clustering(dfs, columns, option):
    if option == "DBSCAN":
        # Max distances between samples, number of samples for core point
        dens = DBSCAN(eps=0.5, min_samples=5)
        dens.fit(dfs)
    else:
        # Number of samples for core point, distance type, min number of samples for a cluster
        dens = OPTICS(min_samples=5, p=2, min_cluster_size=None)
        dens.fit(dfs)

    c_scores = get_scores(dfs, dens.labels_)

    dfs.plot.scatter(x=columns[0], y=columns[1], c=dens.labels_, cmap='rainbow')

    plt.title("Density-Based Clustering with {}".format(option))
    plt.text(8, 18, c_scores)
    plt.show()

    print("\nScores for {}:\n ".format(option) + c_scores)


all_data = prepare_data()

# Accessing the first dataframe, and 2 columns to use
test_columns = ["Emin Medial", "Emin Lateral"]
test_data = all_data[0][test_columns]

# # Run clustering algorithms, plot respective clusters and print scores
kmeans_clustering(test_data, test_columns, 10, 10, 300)
hierarchical_clustering(test_data, test_columns, 10, "ward")
density_clustering(test_data, test_columns, "DBSCAN")
density_clustering(test_data, test_columns, "OPTICS")
