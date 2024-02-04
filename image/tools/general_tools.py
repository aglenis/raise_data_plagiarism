import json
import os
import pickle
from datetime import timedelta
from functools import wraps
from time import time
from typing import Any, Iterable, Dict
from urllib.parse import urlparse

import numpy as np
from numpy import bool_


def get_folder_path(path_from_module: str) -> str:
    """Method to find the folders that in many cases is needed but are not visible.
    Args:
        path_from_module (str): the path from the central repo to the folder

    Returns:
        str
    """

    # Important: We don't need the real path

    fn = os.path.join(os.getcwd(), path_from_module)

    return fn


def get_filepath(path_from_module: str, file_name: str) -> str:
    """Method to find the path-files that in many cases is needed but are not visible.
    Args:
        path_from_module (str): the path from the central repo to the folder
        file_name (str): the file we want from the folder

    Returns:
        str, the actual path to folder
    """

    # Important: We don't need the real path

    fn = os.path.join(os.getcwd(), path_from_module, file_name)
    return fn


def is_number(value: Any) -> bool:
    """
    Args:
        value (any): given arg

    Returns:
        bool
    """
    try:
        float(value)
        return True

    except (ValueError, TypeError):
        return False


def is_empty(value: Any) -> bool:
    """
    Args:
        value (any): given arg

    Returns:
        bool
    """
    return not bool(value)


def extract_info_from_url(url: str) -> dict:
    """Extracting protocol, hostname, path, params, query, username, password and port  from url given.

    Args:
        url (str): given site url

    Returns:
        dict
    """
    obj = urlparse(url)
    return {
        "protocol": obj.scheme,
        "hostname": obj.hostname,
        "path": obj.path,
        "params": obj.params,
        "query": obj.query,
        "username": obj.username,
        "password": obj.password,
        "port": obj.port,
    }


def time_it(method):  # pragma: no cover
    """Print the runtime of the decorated method"""

    # Required for time_it to also work with other decorators.
    @wraps(method)
    def timed(*args, **kwargs):
        start = time()
        result = method(*args, **kwargs)
        finish = time()

        print(
            f"Execution completed in {timedelta(seconds=round(finish - start))} s. "
            f"[method: <{method.__name__}>]"
        )
        return result

    return timed


def load_pickled_data(path: str):  # pragma: no cover
    """Load data from pickle

    Args:
        path (str): path to file

    Returns:
        Pickled Object: data
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File/Dir {path} is not found.")


def dump_pickled_data(path: str, data: object) -> None:  # pragma: no cover
    """Write data to pickle

    Args:
        path (str): path to file
        data (object): data
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    except FileNotFoundError:
        raise FileNotFoundError(f"File/Dir {path} is not found.")


def inverse_iterable(m: Iterable) -> dict:
    """Receives an iterable consisting of pairs and creates an inverse dictionary.
    Args:
        m (Iterable): the iterable to inverse.
    Returns:
        dict: the inverse Dict.
    """
    return {v: k for k, v in m}


def load_json_file(filepath: str) -> Dict:
    """Reads the specified JSON file.

    Args:
        filepath (str): The JSON filepath to read.

    Returns:
        Dict
    """
    try:
        with open(filepath, "r") as json_file:
            return json.load(json_file)

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file{filepath} is not found")


def is_sorted_desc(x: np.array) -> bool_:
    """
    Checks if an array is sorted in descending order
    Args:
        x (np.array): The array that we want to check

    Returns:
        bool
    """
    if len(x) == 0:
        raise ValueError("Array is empty!")

    return (np.diff(x) <= 0).all()
