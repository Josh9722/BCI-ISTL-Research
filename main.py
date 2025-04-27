# ------------- Imports -------------
# External Libs
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws
from tensorflow.keras.models import load_model
import os


# Classes 
from DatasetLoader import DatasetLoader
# from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
from ClusteringModel import ClusteringModel
from AnalyseModels import AnalyseModels



# ------------- Helper Functions -------------

# The original fundction that withholds the complete subject(s)
def split_subjects(epo, val_frac=0.1, test_frac=0.2, seed=40):
    """
    Split an MNE Epochs object into train/val/test with *disjoint subjects*.

    Returns
    -------
    train_epo, val_epo, test_epo
    """
    subj_ids = np.array(sorted(epo.metadata['subject'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subj_ids)

    n_total = len(subj_ids)
    n_test  = max(1, int(n_total * test_frac))
    n_val   = max(1, int(n_total * val_frac))

    test_subj = subj_ids[:n_test]
    val_subj  = subj_ids[n_test:n_test + n_val]
    train_subj = subj_ids[n_test + n_val:]

    mask_test  = epo.metadata['subject'].isin(test_subj)
    mask_val   = epo.metadata['subject'].isin(val_subj)
    mask_train = ~(mask_test | mask_val)

    return epo[mask_train], epo[mask_val], epo[mask_test]


def trainClusteringModel(epochs, trainEpochs = 1, chans = 64, nb_clusters = 3):
    
    clustering_model = ClusteringModel(epochs = epochs, nb_clusters=nb_clusters, embedding_dim=32)
    
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

    
    # clustering_model.plot_clusters(embeddings, cluster_labels) # Optionaly visually inspect the clusters
    return clustering_model


# ------------- Loading Dataset -------------

# Optionally use saved epochs. 
if True:
    print("🔁 Loading preprocessed epochs from disk...")
    epochs = mne.read_epochs("allsubjects-selectedchannels-epo.fif", preload=True)
else:
    print("📥 Generating epochs from raw EEG...")
    loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104], channels = [
    'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',
    'C5..',  'C3..',  'C1..',  'Cz..',  'C2..',  'C4..',  'C6..',
    'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.',
    'P7..',  'P3..',  'Pz..',  'P4..',  'P8..'
    ])
    # loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104])
    loader.load_raw_data()
    loader.epochs.save("allsubjects-selectedchannels-epo.fif", overwrite=True)
    epochs = loader.epochs
    loader.print_event_distribution_by_subject()


print()
print("Data loaded successfully!")



# ------------- Training Model -------------
# Train baseline EEGNet Model
modelName = "EEGNet_Baseline"
trainEpochs = 60
chans = epochs.info['nchan']
saveName = f"{modelName}_{trainEpochs}epochs_{chans}chans.keras"

trainer = ModelTrainer(epochs, modelName)
if os.path.exists(saveName):
    print("Loading existing baseline model...")
    trainer.model = load_model(saveName, safe_mode = False)
else:
    train_epo, val_epo, test_epo = split_subjects(epochs, val_frac=0.1, test_frac=0.2)
    trainer.train(train_epo, val_epo, test_epo, trainEpochs)

    # Save EEGNet model
    trainer.model.save(saveName)
    print("\nModel saved as ", saveName)





# ------------- Clustering Data -------------
# List of clustering models objects 
clustering_models = []

print("\nTraining clustering model...")
model1 = trainClusteringModel(epochs=epochs, trainEpochs = 50, chans = epochs.info['nchan'], nb_clusters = 2)
model2 = trainClusteringModel(epochs=epochs, trainEpochs = 50, chans = epochs.info['nchan'], nb_clusters = 3)
model3 = trainClusteringModel(epochs=epochs, trainEpochs = 50, chans = epochs.info['nchan'], nb_clusters = 4)
model4 = trainClusteringModel(epochs=epochs, trainEpochs = 50, chans = epochs.info['nchan'], nb_clusters = 5)
model5 = trainClusteringModel(epochs=epochs, trainEpochs = 50, chans = epochs.info['nchan'], nb_clusters = 6)

clustering_models.append(model1)
clustering_models.append(model2)
clustering_models.append(model3)
clustering_models.append(model4)
clustering_models.append(model5)

print("Clustering complete!")


# Train model for each cluster
# Get a dictionary mapping cluster labels to lists of subjects

modelIndex = 0
for clustering_model in clustering_models:
    modelIndex += 1
    embeddings, subjects = clustering_model.extract_embeddings()
    cluster_labels = clustering_model.perform_clustering(embeddings)

    clustered_subjects, counts = clustering_model.analyze_clusters_by_subject(cluster_labels, subjects, mode="majority", threshold=0.8, logPath=f"./logs/Cluster Distribution from Model_{modelIndex}", verbose=True)
    clusterNumber = 1

    trainEpochs = 60
    # Iterate over each cluster in the dictionary
    for cluster_label, subject_list in clustered_subjects.items():
        modelName = f"Cluster Model #{modelIndex} Cluster Group #{clusterNumber}"

            
        # Create a boolean mask: True for epochs whose 'subject' is in subject_list
        mask = epochs.metadata['subject'].isin(subject_list)

        # Filter epochs using the boolean mask
        narrowed_epochs = epochs[mask]

        if len(narrowed_epochs) <= 2:
            print(f"Warning: No epochs found for cluster {cluster_label} (#{clusterNumber}).")
            continue

        # (Optional) Print some info to verify the selection
        print(f"Original number of epochs: {len(epochs)}")
        print(f"Number of epochs for subjects in cluster {cluster_label} (#{clusterNumber}): {len(narrowed_epochs)}")
        
        # Train the model using the narrowed epochs
        trainer = ModelTrainer(narrowed_epochs, modelName)
        saveName = f"{modelName}.keras"
        if os.path.exists(saveName):
            print("Loading existing clustered model...")
            trainer.model = load_model(saveName, safe_mode = False)
        else:
            train_epo, val_epo, test_epo = split_subjects(narrowed_epochs, val_frac=0.1, test_frac=0.2)
            trainer.train(train_epo, val_epo, test_epo, trainEpochs)
            trainer.model.save(saveName)
            clusterNumber += 1


# ------------- Analyse Models -------------
baseline_log = "./logs/EEGNet_Baseline_training_log.txt"


# 2 Clusters
clustered_logs = [
    "./logs/Cluster Model #1 Cluster Group #1_training_log.txt",
    "./logs/Cluster Model #1 Cluster Group #2_training_log.txt",
]
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# 3 Clusters
clustered_logs = [

    "./logs/Cluster Model #2 Cluster Group #1_training_log.txt",
    "./logs/Cluster Model #2 Cluster Group #2_training_log.txt",
    "./logs/Cluster Model #2 Cluster Group #3_training_log.txt",
]
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# 4 Clusters
clustered_logs = [
    "./logs/Cluster Model #3 Cluster Group #1_training_log.txt",
    "./logs/Cluster Model #3 Cluster Group #2_training_log.txt",
    "./logs/Cluster Model #3 Cluster Group #3_training_log.txt",
    "./logs/Cluster Model #3 Cluster Group #4_training_log.txt",
]
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# 5 Clusters
clustered_logs = [
    "./logs/Cluster Model #4 Cluster Group #1_training_log.txt",
    "./logs/Cluster Model #4 Cluster Group #2_training_log.txt",
    "./logs/Cluster Model #4 Cluster Group #3_training_log.txt",
    "./logs/Cluster Model #4 Cluster Group #4_training_log.txt",
    "./logs/Cluster Model #4 Cluster Group #5_training_log.txt",
]
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# 6 Clusters
clustered_logs = [
    "./logs/Cluster Model #5 Cluster Group #1_training_log.txt",
    "./logs/Cluster Model #5 Cluster Group #2_training_log.txt",
    "./logs/Cluster Model #5 Cluster Group #3_training_log.txt",
    "./logs/Cluster Model #5 Cluster Group #4_training_log.txt",
    "./logs/Cluster Model #5 Cluster Group #5_training_log.txt",
    "./logs/Cluster Model #5 Cluster Group #6_training_log.txt",
]
analyser = AnalyseModels(baseline_log, clustered_logs)
analyser.produceReport()


# ------------- Testing Model -------------
# model = load_model("eegnet_model.keras")
# tester = ModelTester(model, epochs)
# tester.test(random_state=32)


# ------------- Saving Model ------------- 




