# ------------- Imports -------------
# External Libs
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws


# Classes 
from DatasetLoader import DatasetLoader
# from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
# from ModelTester import ModelTester
from ClusteringModel import ClusteringModel

# ------------- Loading Dataset -------------
# loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12], channels=['Fc5.', 'C3..', 'C4..', 'Cz..'])
loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12])

# Load raw EEG data
loader.load_raw_data()
epochs = loader.epochs

# Optional Visualisation
# loader.epochs.plot()
# plt.show()

print()
print("Data loaded successfully!")


# ------------- PreProcessing Data -------------
#preprocessor = DataPreprocessor()

# ------------- Clustering Data -------------
print("\nTraining clustering model...")
clustering_model = ClusteringModel(epochs, nb_clusters=3, embedding_dim=32)
clustering_model.train_embedding_model(train_epochs=50, batch_size=64)
embeddings, subjects = clustering_model.extract_embeddings()
cluster_labels = clustering_model.perform_clustering(embeddings)
clustering_model.analyze_clusters_by_subject(cluster_labels, subjects)
clustering_model.plot_clusters(embeddings, cluster_labels)
print("Clustering complete!")

# ------------- Training Model -------------
#trainer = ModelTrainer()

#trainer = ModelTrainer(epochs)
#trainer.train()

# ------------- Testing Model -------------
# tester = ModelTester()

# ------------- Saving Model ------------- 



