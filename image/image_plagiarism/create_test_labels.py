import json
import os
from typing import Dict, Tuple, List

from src.tools.config_utils import load_config
from src.tools.general_tools import get_filepath, get_folder_path
from src.image_plagiarism.config import IMAGE_CONFIG


def create_labels(evaluation_dict_path: str = None) -> Tuple[List, Dict]:
    """
    A method that according to a json file creates labels for every query image
    The benchmark has 3 categories for each query: good, ok, and junk
    Good and ok are labeled as 1 while junk as 0
    Args:
        evaluation_dict_path (str): A path to a json file

    Returns:
        list, dict
    """
    config = load_config(IMAGE_CONFIG)
    if evaluation_dict_path is not None:
        images_dict = json.load(open(evaluation_dict_path, 'rb'))
    else:
        path = get_filepath('data/', config['evaluation_dict_path'])
        images_dict = json.load(open(path, 'rb'))

    all_images = []
    for key, value in images_dict.items():
        ims = value["good"] + value["ok"] + value["junk"]
        all_images += ims

    eval_dict = {}

    for key, value in images_dict.items():
        positive_images = value["good"] + value["ok"]

        for query_im in value["query"]:
            eval_dict[query_im] = {im: 1 if im in positive_images else 0 for im in all_images}

    return all_images, eval_dict


if __name__ == '__main__':
    print(create_labels())
