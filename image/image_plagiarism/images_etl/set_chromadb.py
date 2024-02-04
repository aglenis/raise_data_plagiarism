import json
import os
from typing import List

import chromadb
import pandas as pd

from chromadb.api.models.Collection import Collection
from PIL import Image

from src.image_plagiarism.config import IMAGE_CONFIG
from src.tools.config_utils import load_config
from src.tools.evaluation_metrics import calculate_metrics_at_k
from src.tools.general_tools import get_filepath

config = load_config(IMAGE_CONFIG)


class ChromaDB:
    def __init__(self, path_to_save: str, name: str) -> None:
        """
        The constructor of the class. Initializes a client for the database
        and a loads or creates a collection
        Args:
            path_to_save (str): The path to save the database
            name (str): Name of the collection
        """
        self.path_to_save = path_to_save
        self.client = chromadb.PersistentClient(path=self.path_to_save)
        self.collection = self.get_create_collection(name)

    def get_create_collection(self, name: str) -> Collection:
        """
        A method that creates or loads (if exists) a collection according to
        the given name.
        Args:
            name (str): The name of the collection

        Returns:

        """
        collection = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        return collection

    def add_ids_embeddings(self, emb_id: str, embedding: List) -> None:
        """
        A method that adds elements (ids and corresponding embeddings) to a collection
        Args:
            emb_id (str): The id of the embeddings
            embedding (list): The embedding

        Returns:
            None
        """
        self.collection.add(
                    embeddings=embedding,
                    ids=emb_id
                )

    def query_embedding(self, image_embedding: List):
        """
        A method that return the results for the embeddings of the query image
        Args:
            image_embedding (list):

        Returns:

        """
        test_dict = self.collection.query(query_embeddings=image_embedding,
                                          n_results=self.collection.count())
        # we do 1-distance because it returns cosine distance and not cosine similarity
        results_dict = {test_dict["ids"][0][i]: 1-test_dict["distances"][0][i]
                        for i in range(len(test_dict["ids"][0]))}

        return results_dict

