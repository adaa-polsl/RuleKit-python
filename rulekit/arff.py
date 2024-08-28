"""Contains helper functions .arff files.
"""
import pandas as pd
from scipy.io import arff


def read_arff(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Reads an .arff file and returns a pandas DataFrame.

    This function offers a more convenient interface to read .arff files
    than scipy.io arff.loadarff function. It also fix multiple problems with it,
    such as handling missing "?" values and encoding nominal columns.

    Args:
        file_path (str): Path to the .arff file
        encoding (str, optional): Optional file encoding. Defaults to "utf-8".

    Returns:
        pd.DataFrame: pandas DataFrame with the data
    """
    df = pd.DataFrame(arff.loadarff(file_path)[0])
    # code to change encoding of the file
    decoded_df: pd.DataFrame = df.select_dtypes([object])
    decoded_df = decoded_df.stack().str.decode(encoding).unstack()
    for col in decoded_df:
        df[col] = decoded_df[col]
    df = df.where(pd.notnull(df), None)
    df = df.replace({'?': None})
    return df
