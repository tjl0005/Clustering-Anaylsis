from Code.visualise import vis_clusters
from Code.clustering import run_ca
from Code.data import proc_data
from Code.evaluate import sweep
import json

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

method = ["Min-Max", "PCA"]

# Dataframes for differences where method
l_diff = proc_data(["00R", "24R"], method)
r_diff = proc_data(["00R", "24R"], method)

vis_clusters("K-Means", "Left", method, l_diff, run_ca("K-Means", l_diff, params))
vis_clusters("Hierarchical", "Left", method, l_diff, run_ca("Hierarchical", l_diff, params))

vis_clusters("K-Means", "Right", method, r_diff, run_ca("K-Means", r_diff, params))
vis_clusters("Hierarchical", "Right", method, r_diff, run_ca("Hierarchical", r_diff, params))

sweep("K-Means", "Left", l_diff, params, method)
sweep("Hierarchical", "Left", l_diff, params, method)

sweep("K-Means", "Right", r_diff, params, method)
sweep("Hierarchical", "Right", r_diff, params, method)