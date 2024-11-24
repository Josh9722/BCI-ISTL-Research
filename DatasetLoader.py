import mne
from mne.io import concatenate_raws
import numpy as np

class DatasetLoader:
    def __init__(self, subjects=range(1, 80), runs=[4, 8, 12]):
        """
        Initializes the DatasetLoader class.
        
        :param subjects: Range or list of subject IDs to load.
        :param runs: List of motor imagery run IDs.
        """
        self.subjects = subjects
        self.runs = runs
        self.raw = None
        self.events = None
        self.event_id = None
        self.file_paths = []  # Store file paths

    def load_raw_data(self, trim=False):
        """
        Loads, optionally trims, and concatenates raw EEG data for all specified subjects and runs.

        :param trim: Boolean flag to toggle trimming of raw data around events.
        """
        print("Loading dataset...")
        physionet_paths = [mne.datasets.eegbci.load_data(sub_id, self.runs) for sub_id in self.subjects]
        self.file_paths = np.concatenate(physionet_paths)  # Save the file paths

        raw_list = []

        for path in self.file_paths:
            print(f"Processing file: {path}")
            # Load raw EEG data
            raw = mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose='WARNING')

            # Extract events and event IDs
            events, event_id = mne.events_from_annotations(raw)
            print(f"Event mappings for {path}: {event_id}")

            if not events.any():
                print(f"No events found in {path}. Skipping...")
                continue

            raw_list.append(raw)

        if not raw_list:
            raise ValueError("No usable data was found across all files.")

        print("Concatenating raw EEG data...")
        # Concatenate all raw data
        self.raw = mne.concatenate_raws(raw_list)
        print("Concatenation complete. Combined raw data is ready.")


    def filter_by_channels(self, channel_names):
        if self.raw is None:
            raise ValueError("Raw data is not loaded. Please load the raw data before filtering by channels.")

        # Get indices of the specified channels
        channel_indices = mne.pick_channels(self.raw.info['ch_names'], include=channel_names)

        if not channel_indices:
            raise ValueError(f"None of the specified channels {channel_names} are present in the dataset.")

        # Filter raw data to include only the specified channels
        print(f"Filtering dataset to include channels: {channel_names}")
        self.raw.pick_channels(channel_names)
        print("Channel filtering complete.")

    def save_raw_data(self, filename):
        """
        Saves the raw data to a file for later use.
        
        :param filename: Path to save the raw data.
        """
        if self.raw is None:
            raise ValueError("Raw data has not been loaded. Call load_raw_data() first.")
        print(f"Saving raw data to {filename}...")
        self.raw.save(filename, overwrite=True)
