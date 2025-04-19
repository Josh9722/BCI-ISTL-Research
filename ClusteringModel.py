import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense, Lambda
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import Callback
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import tensorflow.keras.backend as K
import pandas as pd

class ClusteringModel:
    def __init__(self, epochs = 1, nb_clusters=3, embedding_dim=32):
        """
        Initializes the ClusteringModel with epochs (which have metadata for subjects),
        a specified number of clusters, and the desired embedding dimension.
        """
        if epochs is None:
            raise ValueError("Error: No epochs provided to ClusteringModel.")
        self.epochs = epochs
        self.nb_clusters = nb_clusters
        self.embedding_dim = embedding_dim
        self.embedding_model = None
        self.clusters = None

    def prepare_data(self):
        """
        Extracts the EEG data and subject metadata.
        Returns:
          - X: EEG data reshaped for CNN input.
          - subjects: Array of subject IDs from epochs metadata.
        """
        X = self.epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        X = X[..., np.newaxis]      # Reshape to (n_epochs, n_channels, n_times, 1)
        # Retrieve subject IDs from metadata (if available)
        if self.epochs.metadata is not None:
            subjects = self.epochs.metadata['subject'].values
        else:
            subjects = None
        return X, subjects

    def build_embedding_model(self, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        """
        Constructs an embedding network based on the EEGNet architecture.
        The network outputs a normalized embedding vector.
        """
        X, _ = self.prepare_data()
        Chans = X.shape[1]
        Samples = X.shape[2]
        inputs = Input(shape=(Chans, Samples, 1))

        # Temporal Convolution
        x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)

        # Depthwise Spatial Filtering
        x = DepthwiseConv2D((Chans, 1), depth_multiplier=D, use_bias=False,
                            depthwise_constraint=max_norm(1.0))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 4))(x)
        x = Dropout(dropoutRate)(x)

        # Separable Convolution
        x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = AveragePooling2D((1, 8))(x)
        x = Dropout(dropoutRate)(x)

        # Flatten and project to the embedding space
        x = Flatten()(x)
        embeddings = Dense(self.embedding_dim, activation=None, name="embedding")(x)
        # Normalize embeddings to unit length
        norm_embeddings = Lambda(lambda t: K.l2_normalize(t, axis=1), name="norm_embedding")(embeddings)

        self.embedding_model = Model(inputs=inputs, outputs=norm_embeddings)
        
        return self.embedding_model

    def train_embedding_model(self, train_epochs=10, batch_size=64):
        """
        Trains the embedding network using TripletSemiHardLoss.
        Uses subject IDs from metadata as labels to form triplets.
        """
        X, subjects = self.prepare_data()
        self.build_embedding_model()
        print("\nTraining embedding model with TripletSemiHardLoss...")
        # Ensure the subject labels are in integer format
        subjects = subjects.astype(np.int32)
        loss_fn = tfa.losses.TripletSemiHardLoss()
        self.embedding_model.compile(optimizer='adam', loss=loss_fn)
        history = self.embedding_model.fit(
            X, subjects,
            epochs=train_epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[CustomLogger()]
        )
        print("\nEmbedding model training complete!")
        return history

    def extract_embeddings(self):
        """
        Uses the trained embedding model to compute embeddings from the EEG data.
        Returns:
          - embeddings: The computed embedding vectors.
          - subjects: The corresponding subject IDs for each epoch.
        """
        X, subjects = self.prepare_data()
        if self.embedding_model is None:
            raise ValueError("Embedding model is not built/trained. Call train_embedding_model first.")
        embeddings = self.embedding_model.predict(X)
        return embeddings, subjects

    def perform_clustering(self, embeddings):
        """
        Applies k-means clustering to the embeddings.
        Returns the cluster labels and prints the silhouette score.
        """
        print("\nPerforming clustering on embeddings...")
        kmeans = KMeans(n_clusters=self.nb_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels)
        print(f"Clustering complete. Silhouette Score: {silhouette:.4f}")
        self.clusters = cluster_labels
        return cluster_labels

    def plot_clusters(self, embeddings, cluster_labels):
        """
        Reduces embeddings to 2D with PCA and plots the clusters.
        """
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='viridis')
        plt.title("Cluster Visualization (PCA Reduction)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()

    def analyze_clusters_by_subject(self, cluster_labels, subjects, mode="unique", threshold=0.8, verbose=True):
        """
        Summarise how subjects are distributed across clusters and (optionally)
        assign each subject to *one* “canonical” cluster.

        Parameters
        ----------
        cluster_labels : array-like, shape (n_epochs,)
            Cluster index for every epoch.
        subjects : array-like, shape (n_epochs,)
            Subject ID for every epoch (same length as cluster_labels).
        mode : {"unique", "majority", "threshold"}, default="unique"
            - "unique"    : original behaviour - a subject appears in every
                            cluster that contains at least one of its epochs.
            - "majority"  : assign a subject **only** to the cluster that holds
                            the largest share of its epochs.
            - "threshold" : assign a subject to a cluster **only if** that cluster
                            contains at least `threshold` (e.g. 0.8 → 80%) of the
                            subject's epochs.  A subject may end up unassigned.
        threshold : float, optional
            Minimum fraction of a subject's epochs that must lie in a cluster for
            "threshold" mode.
        verbose : bool
            Print nice summaries when True.

        """
        import pandas as pd
        df = pd.DataFrame({"cluster": cluster_labels, "subject": subjects})
        # Table of counts: rows = subjects, columns = clusters
        counts = (
            df.groupby(["subject", "cluster"])
            .size()
            .unstack(fill_value=0)
            .astype(int)
            .sort_index()
        )

        if verbose:
            print("\nEpoch counts per subject & cluster:")
            print(counts)

        # ---------- assignment according to mode ----------------------------
        grouped = {cl: [] for cl in counts.columns}  # ensure every cluster key

        if mode == "unique":
            for cl in counts.columns:
                grouped[cl] = counts[counts[cl] > 0].index.tolist()

        elif mode == "majority":
            # pick the cluster with the max count for each subject
            winners = counts.idxmax(axis=1)
            for subj, cl in winners.items():
                grouped[cl].append(subj)

        elif mode == "threshold":
            totals = counts.sum(axis=1)
            frac = counts.div(totals, axis=0)
            for subj in counts.index:
                # clusters that satisfy the threshold for this subject
                good = frac.columns[frac.loc[subj] >= threshold]
                if len(good):
                    # If several clusters pass, pick the one with most epochs
                    best = counts.loc[subj, good].idxmax()
                    grouped[best].append(subj)
                # else: subject remains unassigned

        else:
            raise ValueError("mode must be 'unique', 'majority', or 'threshold'")

        if verbose:
            print("\nSubjects assigned to clusters ({!s} mode):".format(mode))
            for cl, subj_list in grouped.items():
                print(f"Cluster {cl}: {subj_list}")

        return grouped, counts


class CustomLogger(Callback):
    """Custom callback for cleaner training output formatting."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1:02d} | Loss: {logs.get('loss', 0):.4f}")
