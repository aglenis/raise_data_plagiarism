import os
from typing import List, Dict

from src.tools.general_tools import get_filepath


class DataProcessor:
    def __init__(self, evaluation_path: str, images_path: str) -> None:
        """
        The constructor of the class. Create a list with all files
        in the evaluation path and a list with the names of all images
        Args:
            evaluation_path (str): The file of the evaluation files
            images_path (str): The path where all images are stored
        """
        self.evaluation_path = evaluation_path
        self.list_files = os.listdir(self.evaluation_path)
        self.all_images = os.listdir(images_path)

    def create_eval_dict(self) -> Dict:
        """
        A method that creates a dictionary for each query image and its
        corresponding files with good, ok and junk images
        Returns:
            Dict
        """
        queries_files = [file for file in self.list_files if 'query' in file]
        query_file_ids = [query_id.replace('_query.txt', '') for query_id in queries_files]
        queries_files_dict = dict(zip(query_file_ids, queries_files))
        corresponding_files = {self.read_preprocess_queries_txt(value):
                               [i for i in self.list_files if key in i and i != value] for key, value in
                               queries_files_dict.items()}

        return corresponding_files

    def read_preprocess_queries_txt(self, query_file: str) -> str:
        """
        A method the reads and preprocess the queries files.
        They need preprocessing because they have some numbers along
        with the query image. It returns the query image as a string
        Args:
            query_file (str): The name of the query.txt file

        Returns:
            str
        """
        path = get_filepath(self.evaluation_path, query_file)
        with open(path) as f:
            lines = f.read().split(' ')[0]

        # special case for the oxford buildings
        if 'oxc1_' in lines:
            lines = lines.replace('oxc1_', '')
        return lines + '.jpg'

    def read_txt_files(self, txt_file) -> List:
        """
        A method that reads the rest of the files except the queries.txt files.
        It returns a list with all the images in the files
        Args:
            txt_file (str): The name of the txt file

        Returns:
            list
        """
        path = get_filepath(self.evaluation_path, txt_file)
        with open(path) as f:
            lines = f.read().splitlines()
        return [i + '.jpg' for i in lines]

    def create_labels(self) -> Dict:
        """
        A method that creates a dictionary with labels for each query image.
        Specifically it is a nested dictionary with keys the query image and
        values a dictionary with keys every other image and values 0 or 1.
        If the image is in the good or ok txt file of the query image we give
        label 1 else 0
        Returns:
            dict
        """
        corresponding_files = self.create_eval_dict()
        labels_dict = {}

        for key, value in corresponding_files.items():
            labels_dict[key] = {}
            good_images = []
            bad_images = []

            for file in value:
                if "good" in file:
                    good_images += self.read_txt_files(file)
                elif "ok" in file:
                    good_images += self.read_txt_files(file)
                elif "junk" in file:
                    bad_images = self.read_txt_files(file)

            #bad_images = set(self.all_images) - set(good_images)

            #for im in self.all_images:
            for im in good_images:
                labels_dict[key][im] = 1
            for im in bad_images:
                labels_dict[key][im] = 0

        return labels_dict


if __name__ == '__main__':
    a = DataProcessor('/Users/johnmakris/Desktop/data_plagiarism/data/archive/images/oxford_evaluation',
                      '/Users/johnmakris/Desktop/data_plagiarism/data/archive/images/oxford_images')
    print(a.create_labels())
