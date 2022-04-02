from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, OPTICS


def run_ca(c_type, data, params):
    """Produce and display clustering algorithms using specified method"""
    if c_type == "K-Means":
        return kmeans(data, params["kmeans"])
    elif c_type == "Hierarchical":
        return hierarchical(data, params["hierarchical"])
    else:
        return density(data, params[c_type], c_type)


def kmeans(data, k_params):
    """Produce clusters using K-Means"""
    clustering = KMeans(n_clusters=k_params["n_clusters"], n_init=k_params["n_init"], max_iter=k_params["max_iter"])
    clustering.fit(data)

    return clustering


def hierarchical(data, h_params):
    """Produce clusters using Agglomerative Clustering"""
    clustering = AgglomerativeClustering(n_clusters=h_params["n_clusters"], linkage=h_params["linkage"])
    clustering.fit(data)

    return clustering


def density(data, d_params, c_type):
    """Produce clusters using either DBSCAN or OPTICS"""
    # Use different parameters so need to be setup separately
    if c_type == "DBSCAN":
        clustering = DBSCAN(eps=d_params["eps"], min_samples=d_params["min_samples"])
    else:
        clustering = OPTICS(min_samples=d_params["min_samples"], cluster_method=d_params["cluster_method"])

    clustering.fit(data)

    return clustering
