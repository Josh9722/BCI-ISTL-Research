import mne
from mne.io import concatenate_raws
import numpy as np
import pandas as pd 

class DatasetLoader:
    def __init__(self, subjects=range(1, 80), runs=[4, 8, 12], channels=None):
        """
        Initializes the DatasetLoader class.
        
        :param subjects: Range or list of subject IDs to load.
        :param runs: List of motor imagery run IDs.
        :param channels: Optional list of channel names to filter the data.
        """
        self.subjects = subjects
        self.runs = runs
        self.channels = channels  # optional channel filter
        self.epochs = None        # concatenated epochs will be stored here
        self.events = None
        self.event_id = None
        self.file_paths = []      # store file paths

class DatasetLoader:
    def __init__(self, subjects=range(1, 80), runs=[4, 8, 12], exclude_subjects = None, channels=None):
        """
        Initializes the DatasetLoader class.
        
        :param subjects: Range or list of subject IDs to load.
        :param runs: List of motor imagery run IDs.
        :param channels: Optional list of channel names to filter the data.
        """
        self.subjects = subjects
        if exclude_subjects is not None:
            self.subjects = [sub for sub in self.subjects if sub not in exclude_subjects]
            

        self.runs = runs
        self.channels = channels  # optional channel filter
        self.epochs = None        # concatenated epochs will be stored here
        self.events = None
        self.event_id = None
        self.file_paths = []      # store file paths

    def load_raw_data(self, trim=False):
        """
        Loads the EEG data for each subject and each run, extracts epochs from each file,
        and concatenates all epochs into a single Epochs object.
        Boundary annotations (e.g., marking discontinuities) are removed per file before epoching.
        If a list of channels is provided, the epochs are filtered to include only those channels.
        Additionally, attaches metadata for each epoch indicating the subject ID.
        """
        print("Loading dataset for each subject and each run...")
        all_epochs = []  # list to store epochs from each file

        # Iterate over subjects
        for sub_id in self.subjects:
            # Get file paths for the subject's runs
            paths = mne.datasets.eegbci.load_data(sub_id, self.runs, verbose=False)
            self.file_paths.extend(paths)

            # Process each file
            for path in paths:
                raw = mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose=False)

                # Remove boundary annotations (if any)
                boundary_idxs = [i for i, desc in enumerate(raw.annotations.description)
                                if 'boundary' in desc.lower()]
                if boundary_idxs:
                    raw.annotations.delete(boundary_idxs)

                # Extract events and event mapping
                events, event_id = mne.events_from_annotations(raw, verbose=False)

                if not events.any():
                    print(f"No events found in {path}. Skipping...")
                    continue

                # Select EEG channels (this will later be filtered again if needed)
                picks = mne.pick_types(raw.info, eeg=True)

                # Create a metadata DataFrame to attach the subject ID for each epoch
                metadata = pd.DataFrame({'subject': [sub_id] * len(events)})

                # Create epochs from this file (adjust tmin and tmax as needed), including metadata
                epochs = mne.Epochs(raw, events, event_id, tmin=1, tmax=4.1,
                                    proj=False, picks=picks, baseline=None,
                                    preload=True, reject_by_annotation=False,
                                    metadata=metadata, verbose=False)
                print(f"Extracted {len(epochs)} epochs from {path}")
                all_epochs.append(epochs)

        if not all_epochs:
            raise ValueError("No usable epochs were found across all files.")

        # Concatenate epochs from all files
        self.epochs = mne.concatenate_epochs(all_epochs, verbose=False)
        self.events = self.epochs.events
        self.event_id = self.epochs.event_id
        print("Epochs for all runs and subjects have been concatenated.")
        print(f"Total number of epochs: {len(self.epochs)}")

        # If channels are provided, filter the epochs to include only those channels.
        if self.channels is not None:
            print(f"Filtering concatenated epochs to include channels: {self.channels}")
            self.epochs.pick_channels(self.channels)
            print("Channel filtering complete.")

    def filter_by_channels(self, channel_names):
        """
        Additional method to filter the already loaded epochs by a given list of channels.
        """
        if self.epochs is None:
            raise ValueError("Epochs are not loaded. Please call load_raw_data() first.")
        channel_indices = mne.pick_channels(self.epochs.info['ch_names'], include=channel_names)
        if not channel_indices:
            raise ValueError(f"None of the specified channels {channel_names} are present in the dataset.")
        print(f"Filtering epochs to include channels: {channel_names}")
        self.epochs.pick_channels(channel_names)
        print("Channel filtering complete.")

    def save_raw_data(self, filename):
        """
        Saves the concatenated epochs to a file for later use.
        
        :param filename: Path to save the epochs.
        """
        if self.epochs is None:
            raise ValueError("Epochs have not been loaded. Call load_raw_data() first.")
        print(f"Saving epochs to {filename}...")
        self.epochs.save(filename, overwrite=True)



    def save_raw_data(self, filename):
        """
        Saves the raw data to a file for later use.
        
        :param filename: Path to save the raw data.
        """
        if self.raw is None:
            raise ValueError("Raw data has not been loaded. Call load_raw_data() first.")
        print(f"Saving raw data to {filename}...")
        self.raw.save(filename, overwrite=True)
