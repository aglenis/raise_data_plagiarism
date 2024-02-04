import os
from typing import List

import PIL
import torch
import transformers
from PIL import Image
import torchvision.transforms as T
from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor, AutoImageProcessor, AutoModel

from src.image_plagiarism.config import IMAGE_CONFIG
from src.tools.config_utils import load_config
from src.tools.general_tools import get_folder_path


config = load_config(IMAGE_CONFIG)


class VisualEncoder:
    """
    A class with the main functionalities of ViTModel
    """

    def __init__(self, model: transformers.models, extractor: transformers.models,
                 processor: transformers.models) -> None:
        """
        The constructor of the class
        Args:
            model (transformers.models): The model to be used
            extractor (transformers.models): The feature extractor instance.
                                             Should be from the same family as model
            processor (transformers.models): The image processor instance.
                                             Should be from the same family as model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.extractor = extractor
        self.processor = processor

    def image_transformation(self, image: PIL.Image) -> torch.Tensor:
        """
        Method that extracts features from the image by resizing and taking the center
        Args:
            image (PIL.Image): the image to transform

        Returns:
            torch.Tensor
        """
        transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
            ]
        )
        return transformation_chain(image)

    def extract_embeddings(self, image: PIL.Image) -> torch.Tensor:
        """
        A method to extract image embeddings from the last hidden layer of the transformer
        Args:
            image (PIL.Image): the image to extract embeddings

        Returns:
            torch.Tensor
        """

        if not self.extractor:
            print('IM HEREEEE')
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states[:, 0].cpu()

        else:
            image_batch_transformed = self.image_transformation(image)
            pixels = image_batch_transformed.to(self.device)
            num_channels, height, width = pixels.shape
            pixels = pixels.resize(1, num_channels, height, width)
            with torch.no_grad():
                embeddings = self.model(pixels).last_hidden_state[:, 0].cpu()

        return embeddings

    def extract_query_image_embeddings(self, image: str) -> List:
        """
        Method that extracts embeddings for a query image.
        Args:
            image (str): the name of the image

        Returns:
            list
        """
        base_image_path = get_folder_path(config['images_path'])
        image_to_test = Image.open(os.path.join(base_image_path, image))

        query_embeddings = self.extract_embeddings(image_to_test).tolist()
        return query_embeddings
