import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def proc_data(spec, method):
    """Prepare data from specification for clustering"""
    diff = calc_diff(prep(spec[0]), prep(spec[1]))
    diff = dim_reduce(diff, method[0], method[1])

    return diff


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


def dim_reduce(df, scale, reduce):
    """Reduce the dimensions of the given dataframe to 2"""
    # Scale the data before applying reduction techniques
    if scale == "Standard":
        df = StandardScaler().fit_transform(df)
    else:
        df = MinMaxScaler().fit_transform(df)

    # Reduce dimensions by either just PCA or through TSNE using PCA
    if reduce == "PCA":
        df = PCA(n_components=2).fit_transform(df)
    else:
        df = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(df)

    # Return the reduced dataframe
    return pd.DataFrame(data=df, columns=["Component 1", "Component 2"])
