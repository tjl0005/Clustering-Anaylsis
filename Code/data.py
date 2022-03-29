import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)


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
    final_df = pd.concat(dfs, axis=1)

    # Drop all irrelevant columns
    d_columns = ["Year", "LOR", "Error", "Warning", "RatioPixelmm", "Position 10cm Fem"]
    for col in d_columns:
        final_df.drop(col, axis=1, inplace=True)

    # Remove invalid rows
    final_df.replace("", np.nan, inplace=True)
    final_df.dropna(inplace=True)

    # Return without duplicate ID columns
    return final_df.loc[:, ~final_df.columns.duplicated()]


def calc_diff(zero, twenty_four):
    """Find differences between attributes of two dataframes"""
    cols = zero.columns
    df = zero.merge(twenty_four, on="ID")  # Merge years 00 and 24
    ids = df.ID

    # For each column calculate the difference between year 00(x) and year 24(y)
    for col in cols:
        if col != "ID":
            df["{}_diff".format(col)] = df["{}_x".format(col)] - df["{}_y".format(col)]

    # Limit dataframe to only contain difference attributes and reinsert IDs
    df = df.filter(like="diff", axis=1)
    df.insert(loc=0, column="ID", value=ids)

    return df


def reduce(df):
    """Reduce the dimensions of the given dataframe to 2"""
    # ID is not relevant data
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    # Reduce dimensions to n
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)

    # Return the reduced dataframe
    return pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
