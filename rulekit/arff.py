"""Contains helper functions .arff files.
"""
import io
from typing import Union

import numpy as np
import pandas as pd
import requests
from scipy.io import arff


def _is_path_and_url(path: str) -> bool:
    """Checks if the path is a http or https URL.
    """
    return path.startswith("http://") or path.startswith("https://")


def _make_file_object_from_url(url: str) -> io.IOBase:
    """Makes a file-like object from a http or https URL.
    """
    raw_text: str = requests.get(url, timeout=10).text
    return io.StringIO(raw_text)


def read_arff(
    file_path_or_file: Union[str, io.IOBase],
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """Reads an .arff file and returns a pandas DataFrame.

    This function offers a more convenient interface to read .arff files
    than scipy.io arff.loadarff function. It also fix multiple
    problems with it, such as handling missing "?" values and encoding
    nominal columns.

    Args:
        file_path_or_file (str): Either path to the .arff file or a readable
        file-like object. The path can also be a http or https URL.
        encoding (str, optional): Optional file encoding. Defaults to "utf-8".

    Returns:
        pd.DataFrame: pandas DataFrame with the data

    Example:

        >>> # read from file path
        >>> df: pd.DataFrame = read_arff('./cholesterol.arff')
        >>>
        >>> # read from file-like object
        >>> with open('./cholesterol.arff', 'r') as f:
        >>>     df: pd.DataFrame = read_arff(f)
        >>>
        >>> # read from URL
        >>> df: pd.DataFrame = read_arff(
        >>>    'https://raw.githubusercontent.com' +
        >>>    '/adaa-polsl/RuleKit-python/master/tests' +
        >>>    '/additional_resources/cholesterol.arff'
        >>> )
    """
    if (
        isinstance(file_path_or_file, str) and
        _is_path_and_url(file_path_or_file)
    ):
        file_path_or_file = _make_file_object_from_url(file_path_or_file)

    df = pd.DataFrame(arff.loadarff(file_path_or_file)[0])
    # code to change encoding of the file
    decoded_df: pd.DataFrame = df.select_dtypes([np.object_])
    if not decoded_df.empty:
        decoded_df = decoded_df.stack().str.decode(encoding).unstack()
        for col in decoded_df:
            df[col] = decoded_df[col]
    df = df.where(pd.notnull(df), None)
    df = df.replace({'?': None})
    return df
