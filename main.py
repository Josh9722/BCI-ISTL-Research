# ------------- Imports -------------
# External Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Classes 
from DatasetLoader import DatasetLoader
# from DataPreprocessor import DataPreprocessor
# from ModelTrainer import ModelTrainer
# from ModelTester import ModelTester

# ------------- Loading Dataset -------------
loader = DatasetLoader(subjects=range(1, 2), runs=[4])

# Load raw EEG data
loader.load_raw_data()

print()

# Filter by only Fc5 channel 
loader.raw.pick(['Fc5.', 'C3..', 'C4..', 'Cz..'])

# Visualise the raw data given it is of type mne.io.read_raw_edf
loader.raw.plot()
# Visualiser closes straight away, so we need to add a pause
plt.pause(1000)

# ------------- PreProcessing Data -------------
#preprocessor = DataPreprocessor()

# ------------- Training Model -------------
#trainer = ModelTrainer()

# ------------- Testing Model -------------
# tester = ModelTester()

# ------------- Saving Model ------------- 



