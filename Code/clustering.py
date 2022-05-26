from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, AffinityPropagation


def run_ca(c_type, data, params):
    """Produce and display clustering algorithms using specified method"""
    if c_type == "K-Means":
        clustering = KMeans(n_clusters=params["n_clusters"], n_init=params["n_init"], max_iter=params["max_iter"])
    elif c_type == "Hierarchical":
        clustering = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"])
    elif c_type == "Affinity Propagation":
        if params["preference"] != "None":
            clustering = AffinityPropagation(damping=params["damping"], preference=params["preference"], max_iter=1000)
        else:
            clustering = AffinityPropagation(damping=params["damping"], max_iter=1000)
    else:
        clustering = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])

    return clustering.fit(data)


def prod_clusters(types, data, params, vals, param):
    """Produce array of different clusters"""
    c_labels = []

    for i in range(4):
        if isinstance(vals, list):
            params[types][param] = vals[i]
            print(params)
            c_labels.append(run_ca(types, data[i], params[types]))
        else:
            c_labels.append(run_ca(types, data[i], params[types]))
        i += 1

    return c_labels
