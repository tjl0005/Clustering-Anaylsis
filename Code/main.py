from Code.clustering import kmeans_clustering, hierarchical_clustering, density_clustering
from Code.data import diff_calc, prepare_data
from Code.ranking import param_sweep

# Parameters for algorithms
kmeans_params = {
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


# Example function calls
def function_test(diff):
    param_sweep("K-Means", kmeans_params, diff[0], "n_clusters", [4, 5, 6])
    kmeans_clustering(diff[0], kmeans_params, "print")
    # kmeans_clustering(diff[0], k_means_params, "plot")


# Contains a list of dataframes showing the differences of attributes for left and right index knees
left_diff = diff_calc(prepare_data("00L"), prepare_data("24L"))
right_diff = diff_calc(prepare_data("00R"), prepare_data("24R"))

function_test(left_diff)
