import glob
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import FeatureAgglomeration as featAgl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as stdScal, MinMaxScaler as mmScal, RobustScaler as robScal


def initialise(params):
    """Prepare variables to be used for experiments, usually set to default parameters"""
    # Config Directories
    dirs = ["../config/parameters/{}.json".format(params), "../config/methods.json"]

    with open(dirs[0]) as f1, open(dirs[1]) as f2:
        params = json.load(f1)
        method = json.load(f2)

    return vis_scaled(calc_diff(prep("00L"), prep("24L")), "Left", "PCA", "Components")
    # Returns the requested parameters, all methods and the progression profiles
    # return params, method, calc_diff(prep("00L"), prep("24L")), calc_diff(prep("00R"), prep("24R"))


def get_vals(file, spec, param):
    """Get optimised parameter values, requires the algorithm, index and parameter"""
    with open("../config/parameters/{}.json".format(file)) as f:
        vals = json.load(f)
    print(spec)
    return vals[spec][param]


def prep(spec):
    """Create processed dataframe with given specification"""
    # List of all Excel files matching spec
    files = glob.glob("../data/*{}.xls".format(spec))
    ids = pd.read_csv("../data/oai_xrays.csv")
    dfs = []

    # Get ids for relevant index knees
    if spec.endswith("R"):
        ids.drop(ids[ids.side != 1].index, inplace=True)
    else:
        ids.drop(ids[ids.side != 2].index, inplace=True)

    for i in range(len(files)):
        # Add dataframe to list
        df = pd.read_excel(files[i], sheet_name=0)

        # Normalise IDs across dataframes
        df.rename(columns={"Name": "ID"}, inplace=True)
        df["ID"] = pd.to_numeric(df["ID"].replace(".dcm", "", regex=True))

        # Remove rows with irrelevant IDs
        df = df[df["ID"].isin(ids.ID.to_list())]

        dfs.append(df)

    # Turn list of dataframes into a single dataframe
    prep_df = pd.concat(dfs, axis=1)

    # Drop all irrelevant columns
    d_columns = ["Year", "LOR", "Error", "Warning", "RatioPixelmm", "Position 10cm Fem"]
    for col in d_columns:
        prep_df.drop(col, axis=1, inplace=True)

    # Remove invalid rows
    prep_df.replace("", np.nan, inplace=True)
    prep_df.dropna(inplace=True)

    # Return without duplicate ID columns
    return prep_df.loc[:, ~prep_df.columns.duplicated()]


def calc_diff(zero, twenty_four):
    """Find differences between attributes of two dataframes"""
    cols = zero.columns
    diff_df = zero.merge(twenty_four, on="ID")  # Merge years 00 and 24
    ids = diff_df.ID

    # For each column calculate the difference between year 00(x) and year 24(y)
    for col in cols:
        if col != "ID":
            diff_df["{}_diff".format(col)] = diff_df["{}_x".format(col)] - diff_df["{}_y".format(col)]

    # Limit dataframe to only contain difference attributes and reinsert IDs
    diff_df = diff_df.filter(like="diff", axis=1)
    diff_df.insert(loc=0, column="ID", value=ids)

    return diff_df


def proc_data(df, method):
    """Reduce the dimensions of the given dataframe to 2 using specified method"""
    # Scale the data before reducing
    if method["scaling"] == "Standard":
        df = stdScal().fit_transform(df)
    elif method["scaling"] == "Robust":
        df = robScal().fit_transform(df)
    else:
        df = mmScal().fit_transform(df)

    # Reduce dimensions by either just PCA or through TSNE using PCA
    if method["reduction"] == "PCA":
        df = PCA(n_components=2).fit_transform(df)
    else:
        df = featAgl(n_clusters=2).fit_transform(df)

    # Return the reduced dataframe
    return pd.DataFrame(data=df, columns=["Component 1", "Component 2"])


def prod_ds(diff, methods):
    """Produce different versions of the dataset from different processing methods"""
    dfs = []
    if len(methods) == 4:
        for method in methods:
            dfs.append(proc_data(diff, methods[method]))
    else:
        for i in range(4):
            dfs.append(proc_data(diff, methods))

    return dfs


def vis_scaled(df, side, reduction, display):
    """Visualise the effects of scaling and reduction methods"""
    dss = [stdScal().fit_transform(df), robScal().fit_transform(df),
           mmScal().fit_transform(df)]  # Array containing scaled datasets
    s_methods = ["Standard", "Robust", "Min-Max"]  # List of scaling methods for reference
    cols = ["C1", "C2"]
    i = 0

    # Different display methods require different figure configurations
    if display == "Scatter":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
        axes[1, 1].axis('off')
    else:
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), tight_layout=True)

    ax = axes.flatten()  # Store axes as array

    # Go through all scaled versions of the datasets
    while i < 3:
        ax[i].set_title("{} with {} Scaling".format(reduction, s_methods[i]))

        # Apply reduction method
        if reduction == "PCA":
            dss[i] = pd.DataFrame(data=PCA(n_components=2).fit_transform(dss[i]), columns=cols)
        elif reduction == "Feature":
            dss[i] = pd.DataFrame(data=featAgl(n_clusters=2).fit_transform(dss[i]), columns=cols)

        if display == "Scatter":
            ax[i].scatter(dss[i][cols[0]], dss[i][cols[1]])
        else:
            ax[i].plot(dss[i][cols[0]])
            ax[i].plot(dss[i][cols[1]])

        i += 1

    ax[2].annotate('sub2', xy=(0.5, -0.5), va='center', ha='center', weight='bold', fontsize=15)

    # plt.savefig("../Visualisations/{} Indexes/Dataset/{} Indexes Scaled with {}.png".format(side, side, reduction),
    #             dpi=600)
    plt.show()
