import os.path

from src.tools.general_tools import get_filepath

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_CONFIG = get_filepath(ROOT_DIR, 'image_config.toml')
MODEL_CONFIG = get_filepath(ROOT_DIR, 'image_transformer.toml')
CHROMA_CONFIG = get_filepath(ROOT_DIR, 'chromadb.toml')
