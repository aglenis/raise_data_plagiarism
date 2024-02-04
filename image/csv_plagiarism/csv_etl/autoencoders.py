import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from src.tools.data_tools import shuffle_data

# Load and preprocess individual datasets
data1 = pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';')
data1.drop('Target', axis=1, inplace=True)
data2 = pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/TUANDROMD.csv', sep=',')
data2 = data2.iloc[:,:36]
#data3 = pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/MushroomDataset/primary_data.csv', sep=';')

def add_mean_columns(df, num_columns):
    # Calculate the mean of each row
    mean_values = df.mean(axis=1)

    # Create new mean columns dynamically
    mean_columns = {f'Mean_Column_{i+1}': mean_values for i in range(num_columns)}

    # Create a new dataframe with the mean columns
    mean_df = pd.DataFrame(mean_columns)

    # Concatenate the original dataframe and the new mean columns
    df = pd.concat([df, mean_df], axis=1)

    return df

if __name__ == '__main__':
    """
    num_features = np.max([data1.shape[1], data2.shape[1], data3.shape[1]])
    dfs = []
    for data in [data1, data2, data3]:
        if data.shape[1]!=num_features:
            dfs.append(add_mean_columns(data, num_columns=num_features-data.shape[1]))
        else:
            dfs.append(data)

    final_df = pd.concat(dfs, ignore_index=True)
    print(final_df)
    """
    final_data = np.concatenate([data1.values, data2.values])
    print(final_data)
    input_dim = final_data.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    decoded = Dense(input_dim)(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(final_data,
                    final_data,
                    epochs=50, batch_size=32, shuffle=True)

    # Extract embeddings from the encoding layer
    encoder = Model(input_layer, encoded)
    embeddings = encoder.predict(data1)
    shuffled_data = shuffle_data(
        pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';'),
        to_shuffle='columns')
    embeddings_2 = encoder.predict(np.asarray(data2.iloc[:,:36].values).astype(np.float32))#shuffled_data.drop('Target', axis=1))
    from scipy.spatial.distance import cosine
    print(cosine(np.mean(embeddings,axis=0), np.mean(embeddings_2,axis=0)))

"""
# Combine datasets and add a dataset indicator column
data1['Dataset_ID'] = 1
data2['Dataset_ID'] = 2
data3['Dataset_ID'] = 3

combined_data = pd.concat([data1, data2, data3], ignore_index=True)

# Preprocess combined data (handle missing values, scaling, etc.)
# ...

# Split into features and target (assuming no target in this case)
X = combined_data.drop(columns=['Dataset_ID'])

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train autoencoder
input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True)

# Extract embeddings from the encoding layer
encoder = Model(input_layer, encoded)
embeddings = encoder.predict(X_scaled)

# Save or use the learned embeddings for further analysis
"""