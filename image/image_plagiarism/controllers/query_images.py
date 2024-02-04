from typing import Dict

from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor

from src.image_plagiarism.config import CHROMA_CONFIG, MODEL_CONFIG
from src.image_plagiarism.model.image_model import VisualEncoder
from src.image_plagiarism.images_etl.set_chromadb import ChromaDB
from src.tools.config_utils import load_config
from src.tools.general_tools import get_folder_path

db_config = load_config(CHROMA_CONFIG)
chroma_db_class = ChromaDB(get_folder_path(db_config["path_to_save"]), db_config["collection_name"])

model_config = load_config(MODEL_CONFIG)
model = VisualEncoder(eval(model_config['image_encoder']),
                      eval(model_config['image_feature_extractor']),
                      eval(model_config['image_processor']))


def query_image(image: str) -> Dict:
    """
    A method that returns a dictionary with the most similar images to
    the query image along with their cosine distances
    Args:
        image (str): The query image

    Returns:
        Dict
    """
    query_embeddings = model.extract_query_image_embeddings(image)
    test_dict = chroma_db_class.query_embedding(query_embeddings)
    return test_dict


if __name__ == "__main__":
    print(query_image("pitt_rivers_000087.jpg"))
