from data import initialise, prod_ds, get_vals
from clustering import prod_clusters
from visualise import vis_compare
from evaluate import sweep

# Get parameters and methods from config files, and get progression profiles
params, methods, left, right = initialise("default")  # Options are default, optimised left and right
# Select types to be tested, can select individual by using element
types = ["K-Means", "Hierarchical", "Affinity Propagation", "DBSCAN"]

"""Producing default clusters"""
dfs = prod_ds(left, methods["pca_rob"])
clusters = prod_clusters(types[0], dfs, params, params[types[0]], "n_clusters")
vis_compare("left", dfs, clusters, methods["pca_rob"], types[0])

"""Producing optimised clusters"""
vals = get_vals("kmeans", "left", "n_clusters")
dfs = prod_ds(left, methods)
clusters = prod_clusters(types[0], dfs, params, vals, "n_clusters")
vis_compare("Left", dfs, clusters, methods, types[0])

"""Parameter sweep"""
sweep(types[3], ["Left", left],  params, methods["pca_rob"])
