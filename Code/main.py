from Code.clustering import kmeans_clustering, hierarchical_clustering, density_clustering
from Code.data import diff_calc, prepare_data
from Code.ranking import param_sweep
import json

# Default parameters for the algorithms
f = open("../config/def_params.json")
params = json.load(f)
f.close()


# Example function calls
def function_test(diff):
    param_sweep("K-Means", params['kmeans'], diff[0], "n_clusters", [4, 5, 6])
    kmeans_clustering(diff[0], params['kmeans'], "print")
    # kmeans_clustering(diff[0], params['kmeans'], "plot")


# Contains a list of dataframes showing the differences of attributes for left and right index knees
left_diff = diff_calc(prepare_data("00L"), prepare_data("24L"))
right_diff = diff_calc(prepare_data("00R"), prepare_data("24R"))

function_test(left_diff)
