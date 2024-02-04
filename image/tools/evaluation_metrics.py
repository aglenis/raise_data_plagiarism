from typing import List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.tools.general_tools import is_sorted_desc


def precision_at_k_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> np.ndarray:
    """
    Calculate precision at k for every rank.

    Precision@k is defined as the fraction of relevant
    items present in the top-k recommended:

    prec@k = |recommended && relevant| / |recommended|

    Args:
        y_true (np.ndarray): Array with the true labels. 1 if relevant else 0
        y_pred (np.ndarray): Array with the scores from KGE model

    Returns:
        np.ndarray
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if not is_sorted_desc(y_pred):
        raise ValueError("Predictions should be sorted in descending order..")

    n = len(y_true)

    # sort the predictions
    slc = slice(None, None, -1)

    sort_idx = np.argsort(y_pred)[slc]

    # calculate precision at every position
    prec_at_k = np.zeros(n)

    prec_at_k[sort_idx] = np.cumsum(y_true[sort_idx]) / np.arange(1, n + 1)

    return prec_at_k


def recall_at_k_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> np.ndarray:
    """
    Calculate recall at k for every rank.

    Recall@k is defined as the proportion of the total
    relevant items returned in the top-k recommended:

    recall@k = |recommended && relevant| / |relevant|


    Args:
        y_true (np.ndarray): Array with labels. The labels must be in a {0, 1} format, where 1 denotes relevant items
                             and 0 irrelevant ones.
        y_pred (np.ndarray): Array with scores to use for ranking. Higher score is better.

    Returns:
        np.ndarray
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if not is_sorted_desc(y_pred):
        raise ValueError("Predictions should be sorted in descending order..")

    n = len(y_true)
    n_pos = np.sum(y_true)
    slc = slice(None, None, -1)
    sort_idx = np.argsort(y_pred)[slc]

    recall_at_k = np.zeros(n)
    recall_at_k[sort_idx] = np.cumsum(y_true[sort_idx]) / n_pos

    return recall_at_k


def calculate_metrics_at_k(labels: Union[np.ndarray, List], scores: Union[np.ndarray, List],
                           k_s: List) -> Dict:
    """Calculate metrics @ k.

    Args:
        labels (np.ndarray): 0,1} indicator whether result belong to benchmark.
        scores  (np.ndarray) : KGE scores sorted.
        k_s: A list of k's to calculate metrics at.

    Returns:
        Dict: The metrics @ k results.
    """
    precision_k = precision_at_k_score(y_true=labels, y_pred=scores)

    recall_k = recall_at_k_score(y_true=labels, y_pred=scores)

    results = {}
    for idx in k_s:
        results[f"precision@{idx}"] = (
            float(precision_k[idx-1]) if len(precision_k) >= (idx + 1) else None
        )

        results[f"recall@{idx}"] = (
            float(recall_k[idx-1]) if len(recall_k) >= (idx + 1) else None
        )

        results[f"average_precision@{idx}"] = float(
            average_precision_score(y_true=labels[:idx], y_score=scores[:idx])
            if 1 in labels[:idx]
            else 0,
        )

    return results


def organize_k_metrics_mean(results_list_dict: List[Dict]) -> pd.DataFrame:
    """A method that creates a dataframe with the mean of k_metrics

    Args:
        results_list_dict( list of dictionaries): A list with the k_metrics of each test entity

    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame(results_list_dict)
    return df.mean().to_frame().T
