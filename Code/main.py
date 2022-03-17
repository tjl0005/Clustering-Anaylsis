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
def prepare_data(spec):
    # List of all Excel files
    files = glob.glob("../data/*{}.xls".format(spec))
    ids = pd.read_csv("../data/oai_xrays.csv")
    dfs = []

    # Get relevant ids
    if spec.endswith("R"):
        ids.drop(ids[ids.side != 1].index, inplace=True)
    else:
        ids.drop(ids[ids.side != 2].index, inplace=True)

    # Go through all xls files
    for i in range(len(files)):
        # Add dataframe to list
        df = pd.read_excel(files[i], sheet_name=0)

        # Remove irrelevant columns
        if "Error" in df.columns:
            df.drop("Error", axis=1, inplace=True)
        elif "Warning" in df.columns:
            df.drop("Warning", axis=1, inplace=True)
        elif "RatioPixelmm" in df.columns:
            df.drop("RatioPixelmm", axis=1, inplace=True)

        df.drop("Year", axis=1, inplace=True)
        df.drop("LOR", axis=1, inplace=True)

        df.rename(columns={"Name": "ID"}, inplace=True)
        df["ID"] = pd.to_numeric(df["ID"].replace(".dcm", "", regex=True))

        # Remove rows with irrelevant IDs
        df = df[df["ID"].isin(ids.ID.to_list())]

        dfs.append(df)

    return dfs


# Find differences between attributes
def diff_calc(zero, twenty_four):
    diff_dfs = []
    for i in range(len(zero)):
        df = zero[i].merge(twenty_four[i], on="ID")  # Merge dataframes using ID

        cols = get_columns(df)

        # For each column calculate the difference between year 00 and year 24
        for col in cols:
            df["{}_diff".format(col)] = df["{}_x".format(col)] - df["{}_y".format(col)]

        ids = df.ID
        df = df.filter(like="diff", axis=1)
        df.insert(loc=0, column="ID", value=ids)

        diff_dfs.append(df)

    return diff_dfs


# Find which columns to use
def get_columns(df):
    if "Emin Medial_x" in df.columns or "Emin Medial" in df.columns:
        return ["Emin Medial", "Emin Lateral", "Tibial Thick"]
    elif "FTA_x" in df.columns or "FTA" in df.columns:
        return ["FTA"]
    elif "JLCA_LowestPoint_x" in df.columns or "JLCA_LowestPoint" in df.columns:
        return ["JLCA_LowestPoint", "JLCA_P0726"]
    else:
        return ["MinLatJSW", "MaxLatJSW", "MeanLatJSW", "MinMedJSW", "MeanMedJSW", "MeanJSW", "MinJSW", "MaxJSW"]


def plot_clusters(df, categories, scores, clusters, c_type):
    # Scatter plot with all labelled values representing clusters
    df.plot.scatter(x=categories[0], y=categories[1], c=clusters.labels_, cmap="rainbow")

    # Plot center points of clusters
    if c_type == "K-Means":
        plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], marker="X", c="black")

    plt.title("{} Clusters for {}".format(c_type, categories))
    plt.text(8, 18, scores)  # Show scores in visualisation
    plt.show()


def get_scores(data, labels, c_type, display):
    # A measure of how dense and seperated the clusters are
    silhouette = np.round(metrics.silhouette_score(data, labels), 3)  # 1 is best, 0 worst
    # Measure of separation
    davies_bouldin = np.round(metrics.davies_bouldin_score(data, labels), 3)  # Lower is better
    # How effective the number of clusters are
    calinski_harabasz = np.round(metrics.calinski_harabasz_score(data, labels), 3)  # Higher is better

    if silhouette == -1 or calinski_harabasz == -1 or davies_bouldin == -1:
        print("Score error")
    elif display:
        return ("Scores for {}:\n Silhouette Coefficient: {}\n Calinski Harabasz: {}\n Davies-Bouldin Index: {}\n"
                .format(c_type, silhouette, calinski_harabasz, davies_bouldin))
    else:
        return silhouette, calinski_harabasz, davies_bouldin


def param_sweep(c_type, c_params, data, categories, param, vals):
    silhouette = []
    davies = []
    calinski = []

    # Go through all potential parameter values
    for val in vals:
        # Update specified parameter value
        c_params[param] = val

        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            data, labels = kmeans_clustering(data, categories, c_params, "score")
        elif c_type == "Hierarchical":
            data, labels = hierarchical_clustering(data, categories, c_params, "score")
        else:
            data, labels = density_clustering(data, categories, c_params, c_type, "score")

        scores = get_scores(data, labels, c_type, "")

        silhouette.append(scores[0])
        calinski.append(scores[1])
        davies.append(scores[2])

    rank_results(silhouette, vals, "Silhouette")
    rank_results(calinski, vals, "Calinski")
    rank_results(davies, vals, "Davies")


def rank_results(scores, vals, metric):
    s_scores = np.argsort(scores)

    print("\n{}:".format(metric))

    if metric == "Calinski":
        for i in s_scores:
            print("Score: {} from value: {}".format(scores[i], vals[i]))
    else:
        for i in reversed(s_scores):
            print("Score: {} from value: {}".format(scores[i], vals[i]))


def kmeans_clustering(data, categories, k_params, display):
    k_means = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    k_means.fit(data)

    # Decide how to display results
    if display == "plot":
        plot_clusters(data, categories, get_scores(data, k_means.labels_, "K-Means", True), k_means, "K-Means")
    elif display == "print":
        print(get_scores(data, k_means.labels_, "K-Means", True))
    else:
        return data, k_means.labels_


def hierarchical_clustering(data, categories, h_params, display):
    hierarchical = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    hierarchical.fit(data)

    if display == "plot":
        plot_clusters(data, categories, get_scores(data, hierarchical.labels_, "Hierarchical", True), hierarchical,
                      "Hierarchical")
    elif display == "print":
        print(get_scores(data, hierarchical.labels_, "Hierarchical", True))
    else:
        return data, hierarchical.labels_


def density_clustering(data, categories, d_params, c_type, display):
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        db_c = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        db_c = OPTICS(min_samples=d_params["min_samples"], p=d_params["p"],
                      min_cluster_size=d_params["min_cluster_size"])

    db_c.fit(data)

    if display == "plot":
        plot_clusters(data, categories, get_scores(data, db_c.labels_, c_type, True), db_c, c_type)
    elif display == "print":
        print(get_scores(data, db_c.labels_, c_type, True))
    else:
        return get_scores(data, db_c.labels_, c_type, False)


# Contains a list of dataframes showing the differences of attributes
left_diff = diff_calc(prepare_data("00L"), prepare_data("24L"))
right_diff = diff_calc(prepare_data("00R"), prepare_data("24R"))
