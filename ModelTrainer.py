import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import Callback  # ✅ Fix: Import Callback
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt  # ✅ Add this import


class ModelTrainer:
    def __init__(self, epochs):
        if epochs is None:
            raise ValueError("Error: No epochs provided to ModelTrainer.")
        self.epochs = epochs
        self.model = None

    def prepare_data(self):
        """Extracts features from epochs and reshapes data for CNN input."""
        X = self.epochs.get_data()  # shape: (n_samples, n_channels, n_times)
        X = X[..., np.newaxis]  # Reshape to (n_samples, n_channels, n_times, 1) for CNN
        y = self.epochs.events[:, 2]  # Labels from event annotations
        
        # Convert labels from {1, 2, 3} → {0, 1, 2} for TensorFlow compatibility
        y = y - 1  

        # 🚀 Remove one-hot encoding (if previously used)
        # ✅ Make sure `y` remains integers, not categorical

        # Compute class weights
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights = {i: w for i, w in enumerate(class_weights)}

        print(f"\n📊 Computed Class Weights: {class_weights}")
        return X, y, class_weights




    def build_model(self, nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        """ Constructs the EEGNet model. """
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

        # Fully Connected Layer
        x = Flatten()(x)
        outputs = Dense(nb_classes, activation='softmax', kernel_constraint=max_norm(norm_rate))(x)
        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile Model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



    def train(self, epochs=80, batch_size=64, validation_split=0.3):
        """ Trains the EEGNet model with improved logging and per-class accuracy. """
        X, y, class_weights = self.prepare_data()

        # Build EEGNet model
        self.build_model(nb_classes=3, Chans=X.shape[1], Samples=X.shape[2])

        print("\n🚀 Starting Training 🚀")
        history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size,
            class_weight=class_weights, validation_split=validation_split,
            verbose=0,  # Disable default Keras output
            callbacks=[CustomLogger()]
        )

        print("\n✅ Training Complete! Evaluating Performance...")

        # Split data for evaluation
        val_size = int(validation_split * len(y))
        X_test, y_test = X[:val_size], y[:val_size]

        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        # Print per-class accuracy and detailed report
        class_labels = ["T0 (Rest)", "T1 (Left-Hand)", "T2 (Right-Hand)"]
        print("\n📊 Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=class_labels))

        # ✅ Plot Training Curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label="Train Loss")
        plt.plot(history.history['val_loss'], label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label="Train Accuracy")
        plt.plot(history.history['val_accuracy'], label="Val Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")

        plt.show()


    def predict(self, new_epochs):
        """ Predict labels for new epoched data. """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_new = new_epochs.get_data()[..., np.newaxis]  # Reshape for CNN input
        return self.model.predict(X_new)


def focal_loss(alpha=0.5, gamma=2.0):
        """Custom Focal Loss implementation to replace TensorFlow Addons."""
        def loss(y_true, y_pred):
            y_true = K.cast(y_true, tf.float32)
            y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * K.log(y_pred)
            weight = alpha * K.pow(1 - y_pred, gamma)
            return K.mean(weight * cross_entropy, axis=-1)
        return loss

class CustomLogger(Callback):
    """ Custom callback for cleaner output formatting. """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1:02d}/{self.params['epochs']} | "
            f"Loss: {logs.get('loss', 0):.4f} | "
            f"Acc: {logs.get('accuracy', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | "
            f"Val Acc: {logs.get('val_accuracy', 0):.4f}")