import glob
import pandas as pd


# Create dataframes for given specification
def prepare_data(spec):
    # List of all Excel files
    files = glob.glob("../data/*{}.xls".format(spec))
    ids = pd.read_csv("../data/oai_xrays.csv")
    dfs = []

    # Get ids for selected index knee
    if spec.endswith("R"):
        ids.drop(ids[ids.side != 1].index, inplace=True)
    else:
        ids.drop(ids[ids.side != 2].index, inplace=True)

    # Go through the data files
    for i in range(len(files)):
        # Add dataframe to list
        df = pd.read_excel(files[i], sheet_name=0)

        # Remove irrelevant columns
        if "Error" in df.columns:
            df.drop("Error", axis=1, inplace=True)
        elif "Warning" in df.columns:
            df.drop("Warning", axis=1, inplace=True)
        elif "RatioPixelmm" in df.columns:
            df.drop("RatioPixelmm", axis=1, inplace=True)

        # Drop irrelevant columns
        df.drop("Year", axis=1, inplace=True)
        df.drop("LOR", axis=1, inplace=True)
        # Normalise IDs across dataframes
        df.rename(columns={"Name": "ID"}, inplace=True)
        df["ID"] = pd.to_numeric(df["ID"].replace(".dcm", "", regex=True))

        # Remove rows with irrelevant IDs
        df = df[df["ID"].isin(ids.ID.to_list())]

        dfs.append(df)

    return dfs


# Find differences between attributes
def diff_calc(zero, twenty_four):
    diff_dfs = []

    # For each dataframe in year 00
    for i in range(len(zero)):
        df = zero[i].merge(twenty_four[i], on="ID")  # Merge years 00 and 24

        # Select correct list of columns
        if "Emin Medial_x" in df.columns:
            cols = ["Emin Medial", "Emin Lateral", "Tibial Thick"]
        elif "FTA_x" in df.columns:
            cols = ["FTA"]
        elif "JLCA_LowestPoint_x" in df.columns:
            cols = ["JLCA_LowestPoint", "JLCA_P0726"]
        else:
            cols = ["MinLatJSW", "MaxLatJSW", "MeanLatJSW", "MinMedJSW", "MeanMedJSW", "MeanJSW", "MinJSW", "MaxJSW"]

        # For each column calculate the difference between year 00(x) and year 24(y)
        for col in cols:
            df["{}_diff".format(col)] = df["{}_x".format(col)] - df["{}_y".format(col)]

        # Limit dataframe to only contain difference attributes and the original ID
        ids = df.ID
        df = df.filter(like="diff", axis=1)
        df.insert(loc=0, column="ID", value=ids)

        diff_dfs.append(df)

    return diff_dfs
