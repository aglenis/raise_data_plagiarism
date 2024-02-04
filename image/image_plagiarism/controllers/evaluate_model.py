import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor, AutoImageProcessor, AutoModel

from src.image_plagiarism.config import IMAGE_CONFIG, MODEL_CONFIG, CHROMA_CONFIG
from src.image_plagiarism.images_etl.data_processing import DataProcessor
from src.image_plagiarism.model.image_model import VisualEncoder
from src.image_plagiarism.images_etl.set_chromadb import ChromaDB
from src.tools.config_utils import load_config
from src.tools.evaluation_metrics import calculate_metrics_at_k
from src.tools.general_tools import get_folder_path

image_config = load_config(IMAGE_CONFIG)
model_config = load_config(MODEL_CONFIG)
db_config = load_config(CHROMA_CONFIG)
chroma_db_class = ChromaDB(get_folder_path(db_config["path_to_save"]), db_config["collection_name"])
model = VisualEncoder(eval(model_config['image_encoder']),
                      eval(model_config['image_feature_extractor']),
                      eval(model_config['image_processor']))


def model_evaluation():
    eval_metrics = {}
    accuracy_metrics = {}
    data_class = DataProcessor(get_folder_path(image_config["evaluation_dict_path"]),
                               get_folder_path(image_config["images_path"]))
    evaluation_dictionary = data_class.create_labels()
    for image in evaluation_dictionary.keys():
        query_embeddings = model.extract_query_image_embeddings(image)
        test_df = pd.DataFrame(chroma_db_class.query_embedding(query_embeddings).items(),
                               columns=['image', 'predictions'])

        # results_df = chroma_db_class.model_evaluation(evaluation_dictionary)
        test_df['labels'] = test_df['image'].apply(lambda x: evaluation_dictionary[image][x]
        if x in evaluation_dictionary[image].keys() else -1)

        test_df = test_df.loc[test_df['labels'] != -1]
        eval_metrics[image] = calculate_metrics_at_k(test_df['labels'],
                                                     test_df['predictions'], [1, 5, 10])
        threshold_metrics = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            test_df['preds'] = test_df['predictions'].apply(lambda x: 1 if x > threshold else 0)
            threshold_metrics[threshold] = f1_score(test_df.labels, test_df.preds)

        accuracy_metrics[image] = threshold_metrics

    df = pd.DataFrame.from_dict(eval_metrics, orient='index')
    df_accuracy = pd.DataFrame.from_dict(accuracy_metrics, orient='index')
    df.to_csv('/Users/johnmakris/Desktop/data_plagiarism/results/evaluation_results_oxford_deit.csv')
    df_accuracy.to_csv('/Users/johnmakris/Desktop/data_plagiarism/results/classification_results_oxford_deit.csv')

    return df


if __name__ == '__main__':
    print(model_evaluation())
