from Code.clustering import kmeans, hierarchical, density, scores
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def algorithm_sweep(c_type, df, params):
    """"Perform all parameter sweeps for a specified algorithm"""
    print("Results for: {}".format(c_type))

    if c_type == "K-Means":
        param_sweep("K-Means", params["kmeans"], df, "n_clusters", range(2, 11))
        param_sweep("K-Means", params["kmeans"], df, "n_init", range(10, 60, 10))
        param_sweep("K-Means", params["kmeans"], df, "max_iter", range(100, 1100, 100))

    elif c_type == "Hierarchical":
        param_sweep("Hierarchical", params["hierarchical"], df, "n_clusters", range(2, 11))
        param_sweep("Hierarchical", params["hierarchical"], df, "linkage", ["ward", "complete", "average"])

        if params["hierarchical"]["linkage"] != "ward":  # If using ward default affinity must be used
            param_sweep("Hierarchical", params["hierarchical"], df, "affinity", ["euclidean", "manhattan", "cosine"])

    elif c_type == "DBSCAN":
        param_sweep("DBSCAN", params["dbscan"], df, "eps", [0.5, 1, 1.5, 2])
        param_sweep("DBSCAN", params["dbscan"], df, "min_samples", range(0, 10))

    else:
        param_sweep("OPTICS", params["optics"], df, "min_samples", range(5, 12))
        param_sweep("OPTICS", params["optics"], df, "metric", ["cityblock", "cosine", "euclidean", "manhattan"])


def param_sweep(c_type, c_params, data, param, vals):
    """Test algorithm with different parameter values"""
    silhouette = []
    davies = []
    calinski = []
    default = c_params[param]

    for val in vals:
        # Update specified parameter value
        c_params[param] = val

        # Run the correct algorithm with updates parameters and do not produce visualisation
        if c_type == "K-Means":
            data, labels = kmeans(data, c_params, "score")
        elif c_type == "Hierarchical":
            data, labels = hierarchical(data, c_params, "score")
        else:
            data, labels = density(data, c_params, c_type, "score")

        # Store scores for use
        score = scores(data, labels)
        silhouette.append(score[0])
        davies.append(score[1])
        calinski.append(score[2])

    # Create array using the scores
    c_scores = [silhouette, davies, calinski]
    # Reset parameter
    c_params[param] = default
    # Remove current parameter details
    c_params = [x for x in list(c_params.items()) if param not in x]

    # Display results in plot and table
    plot_results(c_scores, vals, param, c_params, c_type)
    print_results(c_scores, vals, param)


def plot_results(score, vals, param, c_params, c_type):
    """Produce subplot of scores for a parameter sweep"""
    fig, ax = plt.subplots(2, 2, tight_layout=True)

    if isinstance(vals[0], int):
        ax[0, 0].grid(alpha=0.5)
        ax[0, 0].plot(vals, score[0], 'o--', c="tab:purple")

        ax[0, 1].grid(alpha=0.5)
        ax[0, 1].plot(vals, score[1], 'o--', c="tab:green")

        ax[1, 0].grid(alpha=0.5)
        ax[1, 0].plot(vals, score[2], 'o--', c="tab:orange")

    else:
        ax[0, 0].grid(alpha=0.5, axis='y')
        ax[0, 0].bar(vals, score[0], width=0.5, color="tab:purple")

        ax[0, 1].grid(alpha=0.5, axis='y')
        ax[0, 1].bar(vals, score[1], width=0.5, color="tab:green")

        ax[1, 0].grid(alpha=0.5, axis='y')
        ax[1, 0].bar(vals, score[2], width=0.5, color="tab:orange")

    # Silhouette Score
    ax[0, 0].set_title("Silhouette Coefficient")
    ax[0, 0].set_ylabel("Score")
    ax[0, 0].set_xlabel(param)

    # Davies-Bouldin Score
    ax[0, 1].set_title("Davies-Bouldin")
    ax[0, 1].set_ylabel("Score")
    ax[0, 1].set_xlabel(param)

    # Calinski-Harabasz Score
    ax[1, 0].set_title("Calinski-Harabasz")
    ax[1, 0].set_ylabel("Score")
    ax[1, 0].set_xlabel(param)

    # Use last subplot to show the current setup
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title("{} with Configuration".format(c_type))
    ax[1, 1].text(0.5, 0.8, "{}\n{}".format(c_params[0], c_params[1]),
                  horizontalalignment="center", verticalalignment="center",
                  bbox=dict(facecolor="none", edgecolor="blue"))

    plt.savefig("../Visualisations/Parameter Sweeps/{}/{}.png".format(c_type, param), dpi=600)


def print_results(c_scores, vals, param):
    """Produce table displaying cluster scores"""
    table = PrettyTable()
    table.add_column(param, vals)
    table.add_column("Silhouette", c_scores[0])
    table.add_column("Davies-Bouldin", c_scores[1])
    table.add_column("Calinski-Harabasz", c_scores[2])

    print(table)
