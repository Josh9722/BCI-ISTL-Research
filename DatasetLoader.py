import mne
from mne.io import concatenate_raws
import numpy as np
import pandas as pd 


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

    
    def print_event_distribution_by_subject(self, labels=("T0", "T1", "T2"), verbose: bool = True):
        epochs = self.epochs            # use the loader's epochs
    
        """
        Print and return a table of event counts per subject.

        Output format (also returned as a DataFrame):
            Subject  T0   T1   T2   total
            1        60   60   60   180
            2        62   62   62   186
            ...

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object with 'subject' metadata and events array.
        labels : tuple[str]
            Class labels to include. Any label absent in event_id is ignored.

        Returns
        -------
        pandas.DataFrame
            Table (index = subject) with counts per label and a 'total' column.
        """
        # 1) Map label → numeric event code (skip missing)
        code_map = {lab: epochs.event_id[lab]
                    for lab in labels if lab in epochs.event_id}

        # 2) Build DataFrame with subject & label columns
        subs = epochs.metadata["subject"].values
        events = epochs.events[:, 2]

        data = {
            "subject": subs,
            "event_code": events
        }
        df = pd.DataFrame(data)

        # 3) Count per subject & label
        counts = (
            df.groupby("subject")["event_code"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={v: k for k, v in code_map.items()})
            .reindex(columns=labels, fill_value=0)    # ensure T0,T1,T2 order
        )

        counts["total"] = counts.sum(axis=1)
        counts = counts.astype(int).sort_index()

        # 4) Pretty-print
        print("─ Event distribution per subject ─")
        for subj, row in counts.iterrows():
            parts = [f"{lab}:{row[lab]}" for lab in labels]
            print(f"{subj:>3} :  " + ", ".join(parts))

        return counts
    
    def balance_T0_T1_epochs(self, epochs, t0_label='T0', t1_label='T1', random_state=42):
        return epochs
        """
        Return a new Epochs where the number of T0 events equals the number of T1 events
        by down‑sampling the T0 class. All other event types remain unchanged.

        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs containing at least the event_id keys t0_label and t1_label.
        t0_label : str
            The key in epochs.event_id corresponding to the “Rest” class.
        t1_label : str
            The key in epochs.event_id corresponding to the “Left‑Hand” class.
        random_state : int
            Seed for reproducible down‑sampling.

        Returns
        -------
        mne.Epochs
            A new Epochs object with len(T0) == len(T1).
        """
        # 1) get the numeric event codes
        ev_map = epochs.event_id
        t0_code = ev_map[t0_label]
        t1_code = ev_map[t1_label]

        # 2) find all indices of T0 and T1 in the concatenated events array
        ev_codes = epochs.events[:, 2]
        idx_t0 = np.where(ev_codes == t0_code)[0]
        idx_t1 = np.where(ev_codes == t1_code)[0]

        # 3) down‑sample T0 to match T1 count
        n_t1 = len(idx_t1)
        rng = np.random.default_rng(random_state)
        if len(idx_t0) > n_t1:
            keep_t0 = rng.choice(idx_t0, size=n_t1, replace=False)
        else:
            keep_t0 = idx_t0

        # 4) keep all T1, the sampled T0, and any other classes unchanged
        idx_other = np.where((ev_codes != t0_code) & (ev_codes != t1_code))[0]
        keep_idx = np.concatenate((keep_t0, idx_t1, idx_other))
        keep_idx = np.sort(keep_idx)

        # 5) return a new Epochs instance with only those trials
        return epochs[keep_idx]
    

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

                  # ▲ 1) Band‑pass 8‑30 Hz (motor‑imagery µ/β band)
                raw.filter(l_freq=8., h_freq=30., picks='eeg',
                        fir_design='firwin', verbose=False)

                # ▲ 2) Notch at 50 Hz (change to 60. if you are in a 60‑Hz region)
                raw.notch_filter(freqs=[50], picks='eeg', verbose=False)

                # Apply average referencing
                raw.set_eeg_reference('average', verbose=False)
                
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
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.0,
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

        self.epochs = self.balance_T0_T1_epochs(self.epochs, t0_label='T0', t1_label='T1')
        print(f"Balanced T0/T1: now {len(self.epochs)} total epochs")

        # If channels are provided, filter the epochs to include only those channels.
        if self.channels is not None:
            print(f"Filtering concatenated epochs to include channels: {self.channels}")
            self.epochs.pick_channels(self.channels)
            print("Channel filtering complete.")

        # ─── Per‑subject z‑score normalization ──────────────────────────────
        print("Applying per‑subject channel‑wise z‑score normalization …")
        data = self.epochs.get_data()  # shape (n_epochs, n_chans, n_times)
        subs = self.epochs.metadata['subject'].values
        for subject in np.unique(subs):
            idx = np.where(subs == subject)[0]
            subj_epochs = data[idx]  # (n_subj_epochs, n_chans, n_times)
            # compute mean/std per channel over time & epochs
            mean = subj_epochs.mean(axis=(0, 2), keepdims=True)
            std  = subj_epochs.std (axis=(0, 2), keepdims=True)
            data[idx] = (subj_epochs - mean) / (std + 1e-10)
        # write back into the Epochs object
        self.epochs._data = data
        print("  ✓ normalization complete.")
        # ────────────────────────────────────────────────────────────────────────

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


