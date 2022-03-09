from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

# Parameters for algorithms
k_means_params = {
    "n_clusters": 10,
    "n_init": 10,  # Number of different centroid seeds to use
    "max_iter": 300  # Maximum number of iterations for one run
}
hierarchical_params = {
    "n_clusters": 10,
    "linkage": "ward",  # Decides which points distances are measured between
    "affinity": "euclidean"  # How distance between points is measured, must euclidean if using ward
}
density_params = {
    "eps": 0.5,  # Decides the max distance between points in one neighbourhood
    "min_samples": 5,  # Decides what can be a core point
    "p": 2,  # How distance will be calculated between points
    "min_cluster_size": None
}


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
    
    # Plot center points of clusters 
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    plt.title("{} Clusters for {}".format(c_type, categories))
    plt.text(8, 18, scores)  # Show scores in visualisation
    plt.show()


def kmeans_clustering(dfs, categories, k_params, plot):
    k_means = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    k_means.fit(dfs)

    cluster_scores = get_scores(dfs, k_means.labels_)
    print("Scores for K-Means:\n " + cluster_scores + "\n")

    # Will only plot the clusters if specified in function call
    if plot:
        plot_clusters(dfs, categories, cluster_scores, k_means, "K-Means")


def hierarchical_clustering(dfs, categories, h_params, plot):
    hierarchical = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    hierarchical.fit(dfs)

    cluster_scores = get_scores(dfs, hierarchical.labels_)
    print("Scores for Hierarchical:\n " + cluster_scores + "\n")

    if plot:
        plot_clusters(dfs, categories, cluster_scores, hierarchical, "K-Hierarchical")


def density_clustering(dfs, categories, d_params, c_type, plot):
    if c_type == "DBSCAN":
        db_c = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        db_c = OPTICS(min_samples=d_params["min_samples"], p=d_params["p"],
                      min_cluster_size=d_params["min_cluster_size"])

    db_c.fit(dfs)
    cluster_scores = get_scores(dfs, db_c.labels_)
    print("Scores for {}:\n ".format(c_type) + cluster_scores + "\n")

    if plot:
        plot_clusters(dfs, categories, cluster_scores, db_c, c_type)


def param_sweep(c_type, c_params, data, categories, param, vals):
    # Go through all potential parameter values
    for val in vals:
        print("With {}={}".format(param, val))
        # Update specified parameter value
        c_params[param] = val
        
        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            kmeans_clustering(data, categories, c_params, False)
        elif c_type == "Hierarchical":
            hierarchical_clustering(data, categories, c_params, False)
        elif c_type == "DBSCAN":
            density_clustering(data, categories, c_params, c_type, False)
        else:
            density_clustering(data, categories, c_params, c_type, False)


all_data = prepare_data()

# Accessing the first dataframe, and 2 categories to use
test_categories = ["Emin Medial", "Emin Lateral"]
test_data = all_data[0][test_categories]

# Run clustering algorithms, plot respective clusters and print scores
kmeans_clustering(test_data, test_categories, k_means_params, True)
hierarchical_clustering(test_data, test_categories, hierarchical_params, True)
density_clustering(test_data, test_categories, density_params, "DBSCAN", True)
density_clustering(test_data, test_categories, density_params, "OPTICS", True)

# Sweeping number of clusters, will output scores for each sweep
param_sweep("K-Means", k_means_params, test_data, test_categories, "n_clusters", [4, 5, 6])
param_sweep("Hierarchical", hierarchical_params, test_data, test_categories, "n_clusters", [4, 5, 6])
# Sweeping size of minimum sample, will output scores for each sweep
param_sweep("DBSCAN", density_params, test_data, test_categories, "min_samples", [4, 5, 6])
param_sweep("OPTICS", density_params, test_data, test_categories, "min_samples", [4, 5, 6])
