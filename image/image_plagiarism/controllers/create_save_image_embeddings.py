import os

from PIL import Image
from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor, AutoImageProcessor, AutoModel, DeiTImageProcessor

from src.image_plagiarism.config import CHROMA_CONFIG, IMAGE_CONFIG, MODEL_CONFIG
from src.image_plagiarism.model.image_model import VisualEncoder
from src.image_plagiarism.images_etl.set_chromadb import ChromaDB
from src.tools.config_utils import load_config
from src.tools.general_tools import get_folder_path


config = load_config(IMAGE_CONFIG)
model_config = load_config(MODEL_CONFIG)


def create_images_embeddings(path_to_images: str):
    path = get_folder_path(path_to_images)
    all_images = os.listdir(path)
    images_to_test = [Image.open(os.path.join(path_to_images, i)) for i in all_images]
    images_dict = dict(zip(all_images, images_to_test))
    model = VisualEncoder(eval(model_config['image_encoder']),
                          eval(model_config['image_feature_extractor']),
                          eval(model_config['image_processor']))
    image_embeddings = {key: model.extract_embeddings(value).tolist() for key, value in images_dict.items()}
    return image_embeddings


def save_embeddings_to_chroma(path_to_images: str):
    embeddings_dict = create_images_embeddings(path_to_images)
    db_config = load_config(CHROMA_CONFIG)
    chroma_db_class = ChromaDB(get_folder_path(db_config["path_to_save"]), db_config["collection_name"])
    for key, value in embeddings_dict.items():
        chroma_db_class.add_ids_embeddings(key, value)


if __name__ == '__main__':
    save_embeddings_to_chroma(config["images_path"])
