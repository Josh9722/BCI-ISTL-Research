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
from AnalyseModels import AnalyseModels

# ------------- Loading Dataset -------------
# loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12], channels=['Fc5.', 'C3..', 'C4..', 'Cz..'])

# Optionally use saved epochs. 
if True:
    print("🔁 Loading preprocessed epochs from disk...")
    epochs = mne.read_epochs("allsubjects-selectedchannels-epo.fif", preload=True)
else:
    print("📥 Generating epochs from raw EEG...")
    loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104], channels=['Fc5.', 'C3..', 'C4..', 'Cz..'])
    loader.load_raw_data()
    loader.epochs.save("allsubjects-selectedchannels-epo.fif", overwrite=True)
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
trainEpochs = 50 
clustering_model = ClusteringModel(epochs, nb_clusters=3, embedding_dim=32)
clustering_model.train_embedding_model(train_epochs=trainEpochs, batch_size=64)
embeddings, subjects = clustering_model.extract_embeddings()
cluster_labels = clustering_model.perform_clustering(embeddings)
clustering_model.analyze_clusters_by_subject(cluster_labels, subjects)
clustering_model.plot_clusters(embeddings, cluster_labels)

# Save the clustering model with the name of the model + the number of epochs trained
clustering_model.embedding_model.save(f"clustering_model_{trainEpochs}epochs.keras")
print("Clustering complete!")

# ------------- Training Model -------------
# Train baseline EEGNet Model
modelName = "EEGNet"
trainer = ModelTrainer(epochs, modelName)
trainer.train()

# Save EEGNet model
saveName = f"{modelName}_model.keras"
trainer.model.save(saveName)
print("\nModel saved as ", saveName)


# Train model for each cluster
# Get a dictionary mapping cluster labels to lists of subjects
clustered_subjects = clustering_model.analyze_clusters_by_subject(cluster_labels, subjects)
clusterNumber = 1

# Iterate over each cluster in the dictionary
for cluster_label, subject_list in clustered_subjects.items():
    modelName = f"EEGNet_clustered #{clusterNumber}"
        
    # Create a boolean mask: True for epochs whose 'subject' is in subject_list
    mask = epochs.metadata['subject'].isin(subject_list)

    # Filter epochs using the boolean mask
    narrowed_epochs = epochs[mask]

    # (Optional) Print some info to verify the selection
    print(f"Original number of epochs: {len(epochs)}")
    print(f"Number of epochs for subjects in cluster {cluster_label} (#{clusterNumber}): {len(narrowed_epochs)}")
    
    # Train the model using the narrowed epochs
    trainer = ModelTrainer(narrowed_epochs, modelName)
    trainer.train() 
    
    clusterNumber += 1


# ------------- Analyse Models -------------
baseline_log = "./logs/EEGNet_training_log.txt"
clustered_logs = [
    "./logs/EEGNet_clustered #1_training_log.txt",
    "./logs/EEGNet_clustered #2_training_log.txt",
    "./logs/EEGNet_clustered #3_training_log.txt"
]

# Create an analyser instance and produce the report.
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# ------------- Testing Model -------------
# model = load_model("eegnet_model.keras")
# tester = ModelTester(model, epochs)
# tester.test(random_state=32)


# ------------- Saving Model ------------- 



