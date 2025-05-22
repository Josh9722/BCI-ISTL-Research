import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    SeparableConv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense, 
    Lambda
)
from tensorflow.keras.constraints import max_norm, Constraint
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D

import tensorflow_addons as tfa

from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from CustomLogger import CustomLogger
from ModelTester import ModelTester 



class ModelTrainer:

    # ---------- helper -------------
    def _extract_xy(self, epo):
        X = epo.get_data()[..., np.newaxis]
        y = epo.events[:, 2] - 1            # {1,2,3} -> {0,1,2}
        # shuffle & class‑weights
        X, y = shuffle(X, y, random_state=40)
        cw_vals = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = {i: w for i, w in enumerate(cw_vals)}
        return X, y, class_weights


    def __init__(self, epochs = None, modelName = "EEGNet", log_dir = './logs', type = "EEGNet"):
        self.epochs = epochs
        self.model = None
        self.train_epochs = None
        self.test_epochs = None
        self.val_epochs = None
        self.modelName = modelName
        # Create log file at .\logs
        # Create the logs directory if it doesn't exist
        self.log_dir = log_dir

        
        # build model
        if type == "EEGNet":
            self.build_EEGNET_model(nb_classes=3,
                            Chans=5,
                            Samples=641)
        elif type == "ShallowNet":
            self.build_ShallowConvNet_model(nb_classes=3,
                            Chans=5,
                            Samples=641)
        elif type == "DeepConvNet":
            self.build_DeepConvNet_model(nb_classes=3, Chans=5, Samples=641)    
        
        
        # Compile with macro‑F1 so callbacks have a target
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tfa.metrics.F1Score(num_classes=3,
                                        average='macro',
                                        name='macroF1')]
        )

        self.inital_weights = self.model.get_weights()
        
    def reset_weights(self):
        """ Reset model weights to initial state. """
        self.model.set_weights(self.inital_weights)

    def prepare_data(self, epo=None, shuffle_data=True):
        """
        Extract X / y and balanced class‑weights from an MNE Epochs object.

        Parameters
        ----------
        epo : mne.Epochs or None
            Source of data.  If None, falls back to self.epochs.
        shuffle_data : bool
            Whether to shuffle X and y together (keeps default behaviour).

        Returns
        -------
        X : np.ndarray  (n_samples, n_chans, n_times, 1)
        y : np.ndarray  (n_samples,)
        class_weights : dict  {class_id : weight}
        """
        epo = epo if epo is not None else self.epochs

        # ----- features & labels ---------------------------------------------
        X = epo.get_data()[..., np.newaxis]       # add channel dim for CNN
        y = epo.events[:, 2] - 1                  # {1,2,3} → {0,1,2}

        # ----- balanced class‑weights ----------------------------------------
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = {int(c): w for c, w in zip(classes, weights)}

        # ----- optional shuffle ----------------------------------------------
        if shuffle_data:
            X, y = shuffle(X, y, random_state=40)

        print(f"\nComputed class weights (len={len(y)}): {class_weights}")
        return X, y, class_weights


    def build_DeepConvNet_model(self,
                            nb_classes,
                            Chans,
                            Samples,
                            dropoutRate=0.5,
                            F1=25,
                            D=2,
                            norm_rate=0.25):
        print("Building DeepConvNet model...")

        inputs = Input(shape=(Chans, Samples, 1))

        # Block 1 - Temporal Conv
        x = Conv2D(F1, (1, 5), padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(dropoutRate)(x)

        # Block 2 - Spatial Conv
        x = Conv2D(F1 * D, (Chans, 1), use_bias=False, kernel_constraint=max_norm(1.0))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(dropoutRate)(x)

        # Block 3
        x = Conv2D(F1 * D * 2, (1, 5), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(dropoutRate)(x)

        # Block 4
        x = Conv2D(F1 * D * 2, (1, 5), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(dropoutRate)(x)

        # Classification
        x = Flatten()(x)
        outputs = Dense(nb_classes, activation='softmax', kernel_constraint=max_norm(norm_rate))(x)

        self.model = Model(inputs=inputs, outputs=outputs)



    def build_EEGNET_model(self, nb_classes, Chans, Samples, dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
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

    def build_ShallowConvNet_model(self,
                               nb_classes,
                               Chans,
                               Samples,
                               dropoutRate=0.5,
                               F1=40,
                               kernLength=25,
                               poolLength=75,
                               stride=15):
        print("Building ShallowConvNet model...")
        def square(x):
            return K.square(x)
        def log(x):
            # avoid log(0)
            return K.log(K.clip(x, K.epsilon(), None))

        inputs = Input(shape=(Chans, Samples, 1))

        # 1) Temporal convolution
        x = Conv2D(F1,
                kernel_size=(1, kernLength),
                padding='same',
                use_bias=False)(inputs)

        # 2) Spatial filtering
        x = Conv2D(F1,
                kernel_size=(Chans, 1),
                use_bias=False,
                kernel_constraint=max_norm(2., axis=(0,1,2)))(x)
        x = BatchNormalization()(x)

        # 3) Square non-linearity
        x = Lambda(square)(x)

        # 4) Average pooling + log non-linearity
        x = AveragePooling2D(pool_size=(1, poolLength),
                            strides=(1, stride))(x)
        x = Lambda(log)(x)

        # 5) Dropout
        x = Dropout(dropoutRate)(x)

        # 6) Classification head
        x = Flatten()(x)
        outputs = Dense(nb_classes,
                        activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

    def train(self, train_epo, val_epo, test_epo, num_epochs=50, batch_size=64):
        os.makedirs(self.log_dir, exist_ok=True)
        log_fileName = f"{self.modelName}_training_log.txt"
        self.log_file = os.path.join(self.log_dir, log_fileName)
        
        self.train_epochs = train_epo
        self.test_epochs = test_epo
        self.val_epochs = val_epo

        X_train, y_train, cw  = self._extract_xy(train_epo)
        X_val,   y_val,  _    = self._extract_xy(val_epo)
        X_test,  y_test, _    = self._extract_xy(test_epo)

        y_train = to_categorical(y_train, 3)
        y_val   = to_categorical(y_val,   3)
        y_test  = to_categorical(y_test,  3)

        # elif type == "EEGNET2":



        cbs = [
            # EarlyStopping(monitor='val_macroF1', patience=15,
            #             mode='max', restore_best_weights=True),
            # ReduceLROnPlateau(monitor='val_macroF1', factor=0.5,
            #                 patience=7, mode='max', min_lr=1e-5),
            CustomLogger(self.log_file)
        ]

        history = self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),   # ← subject-wise val
        epochs=num_epochs,
        batch_size=batch_size,
        class_weight=cw,
        verbose=0,
        callbacks=cbs
        )

        # Final audit on completely unseen subjects
        # evaluateModelPerformance(self, X_test, y_test, history)
        print("\nTraining Complete! Evaluating Performance...")
        
        # tester.per_subject_metrics()
        

    def predict(self, new_epochs):
        """ Predict labels for new epoched data. """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_new = new_epochs.get_data()[..., np.newaxis]  # Reshape for CNN input
        return self.model.predict(X_new)

def evaluateModelPerformance(self, X_test, y_test_onehot, history):
    """
    y_test_onehot : shape (n_samples, 3)  -- one‑hot labels
    """
    # model predictions → class indices
    y_pred = np.argmax(self.model.predict(X_test), axis=1)

    # ground‑truth → class indices
    y_true = np.argmax(y_test_onehot, axis=1)

    class_labels = ["T0 (Rest)", "T1 (Left‑Hand)", "T2 (Right‑Hand)"]
    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

    with open(self.log_file, "a", encoding="utf-8") as f:
        f.write("\nClassification Report:\n")
        f.write(report)

