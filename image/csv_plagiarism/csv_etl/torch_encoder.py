import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.tools.data_tools import shuffle_data, add_gaussian_noise_to_dataset

# Assume df_combined is the combined dataframe from the three datasets
data1 = pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';')
data1.drop('Target', axis=1, inplace=True)
data1 = data1.dropna()
data2 = pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/TUANDROMD.csv', sep=',')
data2 = data2.iloc[:,:36]
data2 = data2.dropna()
# Convert the dataframe to a PyTorch tensor
df_combined = np.concatenate([data1.values, data2.values])
data_tensor = torch.tensor(df_combined, dtype=torch.float32)

# Create a DataLoader for the tensor
data_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)

# Define a simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=np.shape(df_combined)[1], out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=np.shape(df_combined)[1])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
autoencoder = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    for batch in data_loader:

        inputs = batch[0]
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Extract embeddings
with torch.no_grad():
    embeddings_1 = autoencoder.encoder(torch.tensor(data1.values, dtype=torch.float32)).numpy()
    shuffled_data = shuffle_data(
        pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';'),
        to_shuffle='rows')
    noise_data = add_gaussian_noise_to_dataset(
        pd.read_csv('/Users/johnmakris/Desktop/data_plagiarism/data/original_data/data.csv', sep=';'),
        mu=0, sigma=0.5, target_variable='Target')
    embeddings_2 = autoencoder.encoder(torch.tensor(shuffled_data.drop('Target', axis=1).values, dtype=torch.float32)).numpy()
    #embeddings_2 = autoencoder.encoder(
    #    torch.tensor(data2.iloc[:,:36].dropna().values, dtype=torch.float32)).numpy()
from scipy.spatial.distance import cosine
print(cosine(np.mean(embeddings_1,axis=1), np.mean(embeddings_2,axis=1)))