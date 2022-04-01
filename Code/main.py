from Code.clustering import run_ca
from Code.data import proc_data
from Code.ranking import algorithm_sweep
import json

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

# Decide scaling and reduction type
# method = ["Min-Max", "TSNE"]
method = ["Standard", "PCA"]

# Dataframes for differences where method
left = proc_data(["00L", "24L"], method)
right = proc_data(["00L", "24L"], method)

run_ca("K-Means", left, params, method)
run_ca("Hierarchical", left, params, method)

algorithm_sweep("K-Means", left, params, method)
algorithm_sweep("Hierarchical", left, params, method)
