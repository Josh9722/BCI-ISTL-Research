# ------------- Imports -------------
# External Libs
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws
from tensorflow.keras.models import load_model
from ModelTester import ModelTester  # if it's in a separate file
import os


# Classes 
from DatasetLoader import DatasetLoader
# from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
from ModelTester import ModelTester
from ClusteringModel import ClusteringModel

# ------------- Loading Dataset -------------
# loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12], channels=['Fc5.', 'C3..', 'C4..', 'Cz..'])

# if saved epochs exists, load them
if os.path.exists("saved_epochs-epo.fif"):
    print("🔁 Loading preprocessed epochs from disk...")
    epochs = mne.read_epochs("saved_epochs-epo.fif", preload=True)
else:
    print("📥 Generating epochs from raw EEG...")
    loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104])
    loader.load_raw_data()
    loader.epochs.save("saved_epochs-epo.fif", overwrite=True)
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
clustering_model = ClusteringModel(epochs, nb_clusters=5, embedding_dim=32)
clustering_model.train_embedding_model(train_epochs=10, batch_size=64)
embeddings, subjects = clustering_model.extract_embeddings()
cluster_labels = clustering_model.perform_clustering(embeddings)
clustering_model.analyze_clusters_by_subject(cluster_labels, subjects)
clustering_model.plot_clusters(embeddings, cluster_labels)

clustering_model.embedding_model.save("embedding_model.keras")
print("Clustering complete!")

# ------------- Training Model -------------
#trainer = ModelTrainer()

# trainer = ModelTrainer(epochs)
# trainer.train()

# ------------- Testing Model -------------
model = load_model("eegnet_model.keras")
tester = ModelTester(model, epochs)
tester.test(random_state=32)


# ------------- Saving Model ------------- 



