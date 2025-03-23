from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

class ModelTester:
    def __init__(self, model, epochs_object):
        """
        :param model: A loaded Keras model object
        :param epochs_object: MNE Epochs object
        """
        self.model = model
        self.epochs = epochs_object

    def prepare_test_data(self, test_size=0.3, random_state=None):
        """Prepare test data by reshuffling and splitting again."""
        X = self.epochs.get_data()[..., np.newaxis]
        y = self.epochs.events[:, 2] - 1  # Convert from {1, 2, 3} to {0, 1, 2}
        
        # Split new test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_test, y_test

    def test(self, random_state=None):
        """Loads the saved model and evaluates it on a new validation split."""
        # Load model
        print(f"\n📦 Loading model")
        
        # Prepare test set
        X_test, y_test = self.prepare_test_data(random_state=random_state)
        print(f"🧪 Testing on {len(y_test)} samples...")

        # Predict
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        # Report
        class_labels = ["T0 (Rest)", "T1 (Left-Hand)", "T2 (Right-Hand)"]
        print("\n📊 New Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=class_labels))
        print("Testing complete!")

