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

# ------------- Loading Dataset -------------
# loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12], channels=['Fc5.', 'C3..', 'C4..', 'Cz..'])
loader = DatasetLoader(subjects=range(1, 60), runs=[4, 8, 12])

# Load raw EEG data
loader.load_raw_data()
# loader.epochs.plot()
# plt.show()
epochs = loader.epochs

print()
print("Data loaded successfully!")


# ------------- PreProcessing Data -------------
#preprocessor = DataPreprocessor()

# ------------- Training Model -------------
#trainer = ModelTrainer()
trainer = ModelTrainer(epochs)
trainer.train()

# ------------- Testing Model -------------
# tester = ModelTester()

# ------------- Saving Model ------------- 



