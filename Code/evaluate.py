from Code.clustering import kmeans, hierarchical, density
from Code.visualise import vis_results, scores
from prettytable import PrettyTable


def sweep(c_type, side, data, params, method):
    """"Perform all parameter sweeps for a specified algorithm"""
    print("Results for: {}".format(c_type))

    if c_type == "K-Means":
        param_sweep(c_type, params["kmeans"], data, method, side, "n_clusters", range(2, 11))
        param_sweep(c_type, params["kmeans"], data, method, side, "n_init", range(10, 60, 10))

    elif c_type == "Hierarchical":
        param_sweep(c_type, params["hierarchical"], data, method, side, "n_clusters", range(2, 11))
        param_sweep(c_type, params["hierarchical"], data, method, side, "linkage", ["ward", "complete", "average"])

        if params["hierarchical"]["linkage"] != "ward":  # If using ward default affinity must be used
            param_sweep(c_type, params["hierarchical"], data, method, side, "affinity",
                        ["euclidean", "manhattan", "cosine"])

    elif c_type == "DBSCAN":
        param_sweep(c_type, params[c_type], data, method, side, "eps", [0.5, 1, 1.5, 2])
        param_sweep(c_type, params[c_type], data, method, side, "min_samples", range(0, 10))

    else:
        param_sweep(c_type, params[c_type], data, method, side, "min_samples", range(5, 11))


def param_sweep(c_type, c_params, data, method, side, param, vals):
    """Test algorithm with different parameter values"""
    silhouette = []
    davies = []
    default = c_params[param]

    for val in vals:
        # Update specified parameter value
        c_params[param] = val

        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            clusters = kmeans(data, c_params)
        elif c_type == "Hierarchical":
            clusters = hierarchical(data, c_params)
        else:
            clusters = density(data, c_params, c_type)

        # Store scores for use
        score = scores(data, clusters.labels_)
        silhouette.append(score[0])
        davies.append(score[1])

    c_scores = [silhouette, davies]
    # Reset parameter
    c_params[param] = default
    # Remove current parameter details
    c_params = [x for x in list(c_params.items()) if param not in x]

    # Display results in plot and table
    vis_results(c_scores, vals, param, c_params, c_type, method, side)
    print_results(c_scores, vals, param)


def print_results(c_scores, vals, param):
    """Produce table displaying cluster scores"""
    table = PrettyTable()
    table.add_column(param, vals)
    table.add_column("Silhouette", c_scores[0])
    table.add_column("Davies-Bouldin", c_scores[1])

    print(table)
