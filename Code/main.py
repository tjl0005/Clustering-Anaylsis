from Code.clustering import run_ca
from Code.data import proc_data
from Code.ranking import algorithm_sweep
import json

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

# Dataframes for differences for index knees projected into 2D
left = proc_data("00L", "24L")
right = proc_data("00R", "24R")

# # Example clustering calls
# run_ca("K-Means", left, params)
# run_ca("Hierarchical", left, params)
# run_ca("DBSCAN", left, params)
# run_ca("OPTICS", left, params)

# Example parameter sweep calls
algorithm_sweep("K-Means", left, params)
algorithm_sweep("Hierarchical", left, params)
# Currently, has problem with sweeps for algorithm and metric
algorithm_sweep("DBSCAN", left, params)
algorithm_sweep("OPTICS", left, params)
