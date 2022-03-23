from Code.clustering import kmeans, hierarchical, density
from Code.data import calc_diff, prep, reduce
from Code.ranking import param_sweep
import json

# Labels fpr each dataset
labels = ["Eminence Thickness", "Femorotibial Angle", "Joint Line Convergence Angle", "Joint Space Width"]

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

# Dataframes for differences for index knees
left_diff = calc_diff(prep("00L"), prep("24L"))
right_diff = calc_diff(prep("00R"), prep("24R"))

# Example data to cluster and label
diff = reduce(left_diff[2])
label = labels[2]

# Example plots
kmeans(diff, params['kmeans'], "plot", label)
hierarchical(diff, params['hierarchical'], "plot", label)
density(diff, params['density'], "DBSCAN", "plot", label)
density(diff, params['density'], "OPTICS", "plot", label)

# Example parameter sweep
param_sweep("Hierarchical", params['hierarchical'], diff, "n_clusters", [4, 5, 6], label)
