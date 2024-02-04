import json
import os

import pandas as pd

from PIL import Image
from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor

from src.image_plagiarism.config import IMAGE_CONFIG, MODEL_CONFIG
from src.image_plagiarism.create_test_labels import create_labels
from src.image_plagiarism.model.image_model import VisualEncoder
from src.tools.config_utils import load_config
from src.tools.evaluation_metrics import calculate_metrics_at_k, organize_k_metrics_mean
from src.tools.general_tools import get_folder_path, get_filepath
from src.image_plagiarism.utils import fetch_similar

images_embeddings = json.load(open(get_filepath('results/', 'image_embeddings.json'), 'rb'))
config = load_config(IMAGE_CONFIG)
model_config = load_config(MODEL_CONFIG)
model = VisualEncoder(eval(model_config['image_encoder']),
                      eval(model_config['image_feature_extractor']),
                      eval(model_config['image_processor']))


def extract_query_image_embeddings(image):
    base_image_path = get_folder_path(config['images_path'])
    image_to_test = Image.open(os.path.join(base_image_path, image))

    query_embeddings = (image, model.extract_embeddings(image_to_test))
    return query_embeddings


def query_image(image):
    query_embeddings = extract_query_image_embeddings(image)
    return {image: fetch_similar(query_embeddings, images_embeddings)}


def evaluation(query_dict):
    eval_metrics = {}
    for image in query_dict.keys():
        test_df = pd.DataFrame(query_image(image)[image].items(), columns=['image', 'predictions'])
        test_df['labels'] = test_df['image'].apply(lambda x: query_dict[image][x])
        eval_metrics[image] = calculate_metrics_at_k(test_df['labels'],
                                                     test_df['predictions'], [1, 5, 10])
    df = pd.DataFrame.from_dict(eval_metrics, orient='index')
    df.to_csv(get_filepath(config["save_evaluation_path"], config["name_of_evaluation_file"]))
    return df

