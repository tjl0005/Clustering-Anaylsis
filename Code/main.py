from Code.visualise import vis_clusters
from Code.clustering import run_ca
from Code.data import proc_data
from Code.evaluate import sweep
import json

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

method = ["Standard", "PCA"]

# Dataframes for differences where method
diff = proc_data(["00L", "24L"], method)

vis_clusters("K-Means", "Left", method, diff, run_ca("K-Means", diff, params))
vis_clusters("Hierarchical", "Left", method, diff, run_ca("Hierarchical", diff, params))


# sweep("K-Means", "Left", diff, params, method)
# sweep("Hierarchical", "Left", diff, params, method)
