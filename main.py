# ------------- Imports -------------
# External Libs
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
from sklearn.model_selection import LeaveOneGroupOut
import logging

# Classes 
from DatasetLoader import DatasetLoader
from ModelTrainer import ModelTrainer
from ClusteringModel import ClusteringModel
from ModelTester import ModelTester


# ------------- Helper Functions -------------
# ------------- LOSO Helper -------------
def get_train_test(epochs, subjects, leave_out_subj, cluster_map=None):
    """
    Build X_train, y_train, X_test, y_test leaving out all epochs
    for `leave_out_subj`. If cluster_map is given, only epochs
    in leave_out_subj's cluster are used in both train & test.
    """
    # boolean masks
    test_mask  = (epochs.metadata['subject'] == leave_out_subj)
    if cluster_map is not None:
        target_cl = cluster_map[leave_out_subj]
        in_cluster = epochs.metadata['subject'].map(cluster_map) == target_cl
        test_mask &= in_cluster
        train_mask = in_cluster & ~test_mask
    else:
        train_mask = ~test_mask

    train_epo = epochs[train_mask]
    test_epo  = epochs[test_mask]
    return train_epo, test_epo


def run_loso_model(epochs, model_name, n_epochs, log_dir="./logs"):
    # Prepare names and labels
    chans     = epochs.info['nchan']
    save_name = f"{model_name}_{n_epochs}epochs_{chans}chans.keras"
    labels = epochs.events[:, 2] - 1
    classes   = np.unique(labels)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{save_name}.txt")
    
    # Create a dedicated logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # Remove any old handlers (if re-running in same session)
    logger.handlers.clear()

    # Add a file handler just for this run
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    # Write header
    header = ["Subject", "OverallAcc"] + [f"Class{c}Acc" for c in classes]
    logger.info("\t".join(header))

    # LOSO split
    logo = LeaveOneGroupOut()
    model_results = {}

    # Train (assuming ModelTrainer already builds & compiles self.model)
    trainer = ModelTrainer(type = "EEGNet", log_dir=log_dir)
    for train_idx, test_idx in logo.split(X=epochs, y=labels, groups=epochs.metadata['subject']):
        # 1) Clear any old model
        K.clear_session()
        trainer.reset_weights()
        
        subj      = epochs.metadata['subject'].iloc[test_idx[0]]
        train_epo = epochs[train_idx]
        test_epo  = epochs[test_idx]

        trainer.modelName = f"{model_name}_S{subj}"
        trainer.train(
            train_epo,  # train set
            test_epo,  # dummy val set
            test_epo,   # test set
            n_epochs    # number of epochs
        )

        
        raw_preds = trainer.predict(test_epo)
        preds     = np.argmax(raw_preds, axis=1) if raw_preds.ndim == 2 else raw_preds
        true_lbls = labels[test_idx]

        overall_acc = (preds == true_lbls).mean()
        class_accs  = [(preds[true_lbls == c] == c).mean() 
                       if np.any(true_lbls == c) else np.nan
                       for c in classes]

        model_results[subj] = {
            "overall": overall_acc,
            **{f"class_{c}": acc for c, acc in zip(classes, class_accs)}
        }

        row = [subj, f"{overall_acc:.4f}"] + [f"{acc:.4f}" for acc in class_accs]
        logger.info("\t".join(map(str, row)))

        # Overall Results Tester
        tester = ModelTester(trainer)
        tester.overall_report()

    # Clean up handlers so future calls start fresh
    for h in logger.handlers:
        logger.removeHandler(h)
        h.close()
       
    print(f"Model LOSO complete. Logs written to {log_path}")    
    return model_results


def trainClusteringModel(epochs, trainEpochs = 1, chans = 64, nb_clusters = 3):
    labels = ['T0', 'T1', 'T2']
    counts = {lab: len(epochs[lab]) for lab in labels if lab in epochs.event_id}
    counts['total'] = sum(counts.values())
    print("Epoch counts by class:", counts)
    
    # Optional: Get only epochs from event T0 (rest) to perform clustering. 
    epochs = epochs['T0']
    labels = ['T0', 'T1', 'T2']
    counts = {lab: len(epochs[lab]) for lab in labels if lab in epochs.event_id}
    counts['total'] = sum(counts.values())
    print("Epoch counts by class:", counts)

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
if False:
    print("🔁 Loading preprocessed epochs from disk...")
    epochs = mne.read_epochs("allsubjects-selectedchannels-epo.fif", preload=True)
else:
    print("📥 Generating epochs from raw EEG...")
    loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104], channels = ['Fc3.', 'C3..', 'Cz..', 'C4..', 'Fc4.'])
    # loader = DatasetLoader(subjects=range(1, 109), runs=[4, 8, 12], exclude_subjects=[88, 92, 100, 104])
    loader.load_raw_data()
    loader.epochs.save("allsubjects-selectedchannels-epo.fif", overwrite=True)
    epochs = loader.epochs
    loader.print_event_distribution_by_subject()

print()
print("Data loaded successfully!")


# ------------- Training Model -------------
# Train baseline EEGNet Model - LOSO 
modelName   = "EEGNet_Baseline"
log_dir     = "./logs/baseline_model"
trainEpochs = 50
baseline_results = run_loso_model(epochs, modelName, trainEpochs, log_dir)


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


# Train LOSO models for each group of each cluster model. 
modelIndex = 0
trainEpochs = 50
log_dir     = "./logs/clustering_models"

for clustering_model in clustering_models:
    modelIndex += 1

    
    cluster_dir = f"{log_dir}/ClusterModel{modelIndex}"
    os.makedirs(cluster_dir, exist_ok=True) 
    # Prepare the file path
    logPath = f"{cluster_dir}/Cluster_Distribution_Model_{modelIndex}.txt"

    # 1. extract & cluster
    embeddings, subjects     = clustering_model.extract_embeddings()
    cluster_labels           = clustering_model.perform_clustering(embeddings)
    clustered_subjects, _    = clustering_model.analyze_clusters_by_subject(
        cluster_labels,
        subjects,
        mode="majority",
        threshold=0.8,
        logPath=logPath,
        verbose=False
    )

    # 2. for each cluster group, run LOSO with your helper
    clusterNumber = 0
    for cluster_label, subject_list in clustered_subjects.items():
        clusterNumber += 1

        group_name = f"ClusterModel{modelIndex}_Group{clusterNumber}"
        
        # mask out only the epochs for subjects in this cluster
        mask            = epochs.metadata['subject'].isin(subject_list)
        narrowed_epochs = epochs[mask]

        if len(narrowed_epochs) <= 2:
            print(f"⚠️ Skipping cluster {cluster_label} (#{clusterNumber}): too few epochs ({len(narrowed_epochs)})")
            continue

        print(f"→ Running LOSO on cluster {cluster_label} (#{clusterNumber}), {len(narrowed_epochs)} epochs")

        # this will:
        #  • train a separate EEGNet per left‐out subject,
        #  • log per‐subject & per‐class accuracies to ./logs/<group_name>_…txt
        #  • return a dict mapping subject → metrics
        results = run_loso_model(
            narrowed_epochs,
            model_name = group_name,
            n_epochs   = trainEpochs,
            log_dir    = cluster_dir
        )




