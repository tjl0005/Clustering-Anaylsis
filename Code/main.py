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


# Create dataframes for given year
def prepare_data(year):
    # List of all Excel files
    files = glob.glob("../data/*{}.xls".format(year))
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
        elif "RatioPixelmm" in dfs[i].columns:
            dfs[i].drop("RatioPixelmm", axis=1, inplace=True)

        dfs[i].rename(columns={'Name': 'ID'}, inplace=True)
        dfs[i].drop("Year", axis=1, inplace=True)
        dfs[i].drop("LOR", axis=1, inplace=True)
        dfs[i]['ID'] = dfs[i]['ID'].replace('.dcm', '', regex=True)

    # All dataframes
    return dfs


def get_scores(data, labels):
    silhouette = metrics.silhouette_score(data, labels)  # 1 is best, 0 worst
    calinski_harabasz = metrics.calinski_harabasz_score(data, labels)  # Higher is better
    davies_bouldin = metrics.davies_bouldin_score(data, labels)  # Lower is better, 0-1

    # Formatted string to show scores
    return (" Silhouette Coefficient: {}\n Calinski-Harabasz Index: {}\n Davies-Bouldin Index: {}"
            .format(np.round(silhouette, 3), np.round(calinski_harabasz, 3), np.round(davies_bouldin, 3)))


def plot_clusters(df, categories, scores, clusters, c_type):
    # Scatter plot with all labelled values representing clusters
    df.plot.scatter(x=categories[0], y=categories[1], c=clusters.labels_, cmap='rainbow')

    # Plot center points of clusters 
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    plt.title("{} Clusters for {}".format(c_type, categories))
    plt.text(8, 18, scores)  # Show scores in visualisation
    plt.show()


def kmeans_clustering(data, categories, k_params, plot):
    k_means = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    k_means.fit(data)

    cluster_scores = get_scores(data, k_means.labels_)
    print("Scores for K-Means:\n{}\n".format(cluster_scores))

    # Will only plot the clusters if specified in function call
    if plot:
        plot_clusters(data, categories, cluster_scores, k_means, "K-Means")


def hierarchical_clustering(data, categories, h_params, plot):
    hierarchical = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    hierarchical.fit(data)

    cluster_scores = get_scores(data, hierarchical.labels_)
    print("Scores for Hierarchical:\n{}\n".format(cluster_scores))

    if plot:
        plot_clusters(data, categories, cluster_scores, hierarchical, "Hierarchical")


def density_clustering(data, categories, d_params, c_type, plot):
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        db_c = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        db_c = OPTICS(min_samples=d_params["min_samples"], p=d_params["p"],
                      min_cluster_size=d_params["min_cluster_size"])

    db_c.fit(data)
    cluster_scores = get_scores(data, db_c.labels_)
    print("Scores for {}:\n{}\n".format(c_type, cluster_scores))

    if plot:
        plot_clusters(data, categories, cluster_scores, db_c, c_type)


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


# Find differences between attributes
def diff_calc(zero, twenty_four):
    diff_dfs = []
    for i in range(len(zero)):
        df = zero[i].merge(twenty_four[i], on="ID")  # Merge dataframes using ID

        # Find which columns to use
        if "Emin Medial_x" in df.columns:
            cols = ["Emin Medial", "Emin Lateral", "Tibial Thick"]
        elif "FTA_x" in df.columns:
            cols = ["FTA"]
        elif "JLCA_LowestPoint_x" in df.columns:
            cols = ["JLCA_LowestPoint", "JLCA_P0726"]
        else:
            cols = ["MinLatJSW", "MaxLatJSW", "MeanLatJSW", "MinMedJSW", "MeanMedJSW", "MeanJSW", "MinJSW", "MaxJSW"]

        # For each column calculate the difference between year 00 and year 24
        for col in cols:
            df['{}_diff'.format(col)] = df['{}_x'.format(col)] - df['{}_y'.format(col)]

        # Produce a list of the difference dataframe
        diff_dfs.append(df.filter(like='diff', axis=1))
        # Include IDs
        diff_dfs[i] = diff_dfs[i].join(df["ID"])

    return diff_dfs


# Contains a list of dataframes showing the differences of attributes
left_diff = diff_calc(prepare_data("00L"), prepare_data("24L"))
right_diff = diff_calc(prepare_data("00R"), prepare_data("24R"))

print(left_diff[0])
