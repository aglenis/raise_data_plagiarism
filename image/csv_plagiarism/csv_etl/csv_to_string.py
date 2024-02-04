import pandas as pd
from src.tools.data_tools import shuffle_data, add_gaussian_noise_to_dataset


class CsvStringTransformer:
    """
    A class that converts a csv to string
    """

    def __init__(self, path_to_file: str) -> None:
        """
        The constructor of the class
        Args:
            path_to_file (str): The path of the csv file

        """
        self.data = pd.read_csv(path_to_file, sep=';')

    def basic_info_to_string(self, data=None) -> str:
        """
        A funtion that extracts the main statistics i.e., shape, mean of every column
        (most frequent value if categorical), the number of different values for every
        column and others. Also, it adds the information of values for every record.
        Returns:
            str
        """


        if data is None:
            data = self.data

        shape = data.shape
        """
        nunique_values = {column: data[column].nunique() for column in data.columns}
        most_common = {column: data[column].mode()[0] for column in data.columns}
        string_csv = f'This is a dataframe with {shape[0]} row and {shape[1]} columns'
        for i, column in enumerate(data.columns):
            string_csv += (f'Column {i} has {nunique_values[column]} unique values and the most common value is'
                           f'{most_common[column]}')
        """
        string_csv = ''
        for i in range(shape[0]):
            string_csv += f'{data.iloc[i,:].values.tolist()}'

        return string_csv


if __name__ == '__main__':
    a = CsvStringTransformer('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv')
    b = a.basic_info_to_string()
    from transformers import pipeline, BartTokenizer
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    import pandas as pd

    # Load pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(b, return_tensors='pt', truncation=True)

    # Get the model output
    with torch.no_grad():
        output = model(**tokens)

    # Use the output embeddings as the representation
    representation = output.last_hidden_state.mean(dim=1).squeeze().numpy()


    shuffled_data = shuffle_data(pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';'),
                                 to_shuffle='columns')
    noise_data = add_gaussian_noise_to_dataset(pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';'),
                                               mu=0, sigma=0.1, target_variable='Target')
    f = a.basic_info_to_string(noise_data)
        #pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/TUANDROMD.csv', sep=';'))
    tokens = tokenizer(f, return_tensors='pt', truncation=True)

    # Get the model output
    with torch.no_grad():
        output = model(**tokens)

    # Use the output embeddings as the representation
    representation_2 = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    from scipy.spatial.distance import cosine
    print(cosine(representation, representation_2))

    """
    generator = pipeline(model='facebook/bart-large',)
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    batch = tok(b, truncation=True)
    emb = generator(batch)
    print(emb)
    """