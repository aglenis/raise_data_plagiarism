from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_dataset(pd_data: pd.DataFrame, target_variable: str):
    """
    Method that does basic preprocessing in a dataset such as:
        - drop duplicates (rows and columns)
        - min max normalization
        - encoding of target variable (if needed)
    Args:
        pd_data (pd.DataFrame): The dataframe to preprocess
        target_variable (str): the column that a ML algorithm will predict (y)

    Returns:
        pd.DataFrame
    """
    dataset = pd_data.drop_duplicates()
    dataset.drop(columns=dataset.columns[dataset.nunique() == 1], inplace=True)
    dataset = dataset.T.drop_duplicates().T
    features = [col for col in dataset.columns if col != target_variable]
    scaler = MinMaxScaler()
    dataset[features] = scaler.fit_transform(dataset[features])
    encoder = LabelEncoder()
    dataset[target_variable] = encoder.fit_transform(dataset[target_variable])

    return dataset


def add_gaussian_noise_to_dataset(pd_data: pd.DataFrame, mu: Union[float, int],
                                  sigma: Union[float, int], target_variable: str):
    """
    Method that add normally distributed noise to a dataset given a specific mean and std
    of the noise. This will not be applied to the target variable. It returns a dataframe with
    noise injected
    Args:
        pd_data (pd.Dataframe): the dataset to add noise
        mu (float or int): the mean of the noise
        sigma (float or int): the std of the noise
        target_variable (str): the column that a ML algorithm will predict (y)

    Returns:
        pd.DataFrame
    """

    features = [col for col in pd_data.columns if col != target_variable]
    noise = np.random.normal(mu, sigma, [len(pd_data), len(features)])
    pd_data[features] += noise

    return pd_data


def shuffle_data(pd_data: pd.DataFrame, to_shuffle: str):
    """
    A method that shuffles the rows or the columns of a dataframe.
    It returns the same dataframe shuffled
    Args:
        pd_data (pd.DataFrame): the dataset to shuffle
        to_shuffle: choose between rows or columns to shuffle

    Returns:
        pd.DataFrame

    Raises:
        ValueError if to_shuffle is not rows or columns
    """
    if to_shuffle not in ['rows', 'columns']:
        raise ValueError("Argument 'to_shuffle' must either 'rows' or 'columns'")
    if to_shuffle == 'rows':
        return pd_data.sample(frac=1)
    elif to_shuffle == 'columns':
        return pd_data.sample(frac=1, axis=1)


def replace_records_with_values_from_dataset(pd_data: pd.DataFrame,
                                             perc_to_replace: float):
    """
    Method that replaces a specific amount of rows in the dataframe, with values
    that already exist in the dataframe. it returns a new dataframe with
    changed values
    Args:
        pd_data:
        perc_to_replace:

    Returns:
        pd.DataFrame
    """
    total_rows = len(pd_data)
    n_rows = int(round(total_rows * perc_to_replace))
    n_cols = len(pd_data.columns)

    def gen_indices():
        column = np.repeat(np.arange(n_cols).reshape(1, -1), repeats=n_rows, axis=0)
        row = np.random.randint(0, total_rows, size=(n_rows, n_cols))
        return row, column

    row, column = gen_indices()
    new_mat = pd_data.values
    to_place = new_mat[row, column]

    row, column = gen_indices()
    new_mat[row, column] = to_place

    new_data = pd.DataFrame(new_mat, columns=pd_data.columns)
    new_data = new_data.astype(pd_data.dtypes.to_dict())

    return new_data


if __name__ == '__main__':
    dropout_dataset = pd.read_csv('../../data/original_data/data.csv', sep=';')
    dropout_new = preprocess_dataset(dropout_dataset, 'Target')
    print(dropout_new)
    """
    noise_iris = add_gaussian_noise_to_dataset(iris_new, 0, 0.1, 'class')
    shuffled_iris = shuffle_data(iris_new, 'rows')
    swap_iris = replace_records_with_values_from_dataset(shuffled_iris, 0.15)
    print(swap_iris.sepal_length.nunique())
    print(iris_new.sepal_length.nunique())
    """