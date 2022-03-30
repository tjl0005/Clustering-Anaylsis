from Code.clustering import kmeans, hierarchical, density
from Code.data import calc_diff, prep, reduce
from Code.ranking import param_sweep
import json

# Default parameters for the algorithms
with open("../config/def_params.json") as f:
    params = json.load(f)

# # Dataframes for differences for index knees
left_diff = calc_diff(prep("00L"), prep("24L"))
right_diff = calc_diff(prep("00R"), prep("24R"))

# Example data to cluster and label
prep_left_diff = reduce(left_diff)

# Example plots
# kmeans(prep_left_diff, params['kmeans'], "plot")
# hierarchical(prep_left_diff, params['hierarchical'], "plot")
# density(prep_left_diff, params['density'], "DBSCAN", "plot")
# density(prep_left_diff, params['density'], "OPTICS", "plot")

# Example parameter sweep
param_sweep("Hierarchical", params['hierarchical'], prep_left_diff, "n_clusters", [4, 5, 6, 7, 8, 9, 10])
