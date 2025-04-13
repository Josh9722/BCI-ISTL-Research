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


print()
print("Data loaded successfully!")


# ------------- Clustering Data -------------
print("\nTraining clustering model...")
trainEpochs = 50 
chans = epochs.info['nchan']
nb_clusters = 3
print ("Number of channels: ", chans)

clustering_model = ClusteringModel(epochs, nb_clusters=nb_clusters, embedding_dim=32)
# If file exists load file instead
modelName = f"clustering_model_{trainEpochs}epochs_{chans}chans_{nb_clusters}clusters.keras"
if os.path.exists(modelName):
    print("Loading existing clustering model...")
    clustering_model.embedding_model = load_model(modelName, safe_mode = False)
else:
    print("Training new clustering model...")
    clustering_model.train_embedding_model(train_epochs=trainEpochs, batch_size=64)
    
    # Save the clustering model with the name of the model + the number of epochs trained
    clustering_model.embedding_model.save(modelName)

embeddings, subjects = clustering_model.extract_embeddings()
cluster_labels = clustering_model.perform_clustering(embeddings)
clustering_model.analyze_clusters_by_subject(cluster_labels, subjects)
clustering_model.plot_clusters(embeddings, cluster_labels)


print("Clustering complete!")



# ------------- Training Model -------------
# Train baseline EEGNet Model
modelName = "EEGNet"
trainEpochs = 50

trainer = ModelTrainer(epochs, modelName)
if os.path.exists(f"baseline_model_{trainEpochs}epochs.keras"):
    print("Loading existing baseline model...")
    trainer.model = load_model(f"baseline_model_{trainEpochs}epochs.keras", safe_mode = False)
else:
    trainer.train(epochs = trainEpochs)

    # Save EEGNet model
    saveName = f"baseline_model_{trainEpochs}epochs.keras"
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



