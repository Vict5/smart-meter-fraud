# data_processing.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.Autoencoder.data_module import SequenceDataModule


class DataProcessor:
    """Manages data loading and preprocessing"""

    def __init__(self, dataset_path, batch_size, test_size, val_size, seed):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def split_data(self, sequences, dataset_name=None):
        """Split data into train, validation, and test sets"""
        y = [int(seq[1]) for seq in sequences]  # Extract labels

        # Check if we should use stratification
        from collections import Counter
        class_counts = Counter(y)
        use_stratify = not any(count < 2 for count in class_counts.values())

        stratify_param = y if use_stratify else None

        train_sequences, test_sequences, _, _ = train_test_split(
            sequences,
            y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stratify_param,
        )
        y = [int(seq[1]) for seq in test_sequences]

        # Check stratification for second split as well
        stratify_param2 = y if use_stratify else None
        if use_stratify:
            from collections import Counter

            class_counts = Counter(y)
            if any(count < 2 for count in class_counts.values()):
                use_stratify = False
                stratify_param2 = None

        val_sequences, test_sequences, _, _ = train_test_split(
            test_sequences,
            y,
            test_size=self.val_size,
            random_state=self.seed,
            stratify=stratify_param2,
        )

        return train_sequences, val_sequences, test_sequences

    def load_and_preprocess_data(
        self, file_path, only_regular=False, for_training=True
    ):
        """
        Load and preprocess the data to make them fit into the neural network

        Loading: Simpli load from the csv the files Labels and the dataset i need

        Preprocessing: First i remove the supply ID from the data, then Because not every sequence is of the same lenght, i truncate/padding them
            to make them reach the average lenght, if the avg lenght is too large i truncate it to a reasenable value
        """

        # ? ==========LOAD==========
        labels_path = os.path.join(self.dataset_path, "LABELS.csv")
        ts_path = os.path.join(self.dataset_path, file_path)
        # LOAD DATASET
        X_train = pd.read_csv(ts_path, encoding="utf-16", sep="\t")
        # LOAD LABELS
        y_train = pd.read_csv(labels_path, encoding="utf-16", sep="\t")

        if only_regular:  # REMOVE REGULAR DATA
            regular_seq = y_train[y_train["CLUSTER"] == 2]
            regular_seq_ids = regular_seq["Supply_ID"].values
            # update the train data with only regular series
            X_train = X_train[X_train["Supply_ID"].isin(regular_seq_ids)]

        # ? ==========PREPROCESSING==========
        FEATURE_COLUMNS = X_train.columns.tolist()[1:]  # Exclude 'Supply_ID'

        sequences = []  # final sequences

        # MAKE ALL SEQUENCES THE SAME LENGTH

        # Calculate average sequence length
        grouped = X_train.groupby("Supply_ID")
        lengths = [len(group) for _, group in grouped]
        avg_len = int(np.mean(lengths))

        # Cap sequence length to prevent memory issues
        MAX_SEQUENCE_LENGTH = 1000  # Maximum reasonable sequence length
        if avg_len > MAX_SEQUENCE_LENGTH:
            sequence_length = MAX_SEQUENCE_LENGTH
        else:
            sequence_length = avg_len

        # PROCESS THE SEQUENCES

        # for each sequence (one for every supply id)
        for supply_id, group in grouped:
            sequence_features = group[FEATURE_COLUMNS].values

            # Get label for this supply_id
            label_row = y_train[y_train.Supply_ID == supply_id]
            label = label_row.iloc[0].CLUSTER

            # Handle sequence length with the capped value
            if len(sequence_features) >= sequence_length:
                sequence = sequence_features[:sequence_length]
            else:
                padding_length = sequence_length - len(sequence_features)
                padding = np.zeros((padding_length, len(FEATURE_COLUMNS)))
                sequence = np.vstack([sequence_features, padding])

            sequences.append((sequence, label, supply_id))

        # Now the sequences are created

        # Split data
        if for_training:
            train_sequences, val_sequences, test_sequences = self.split_data(
                sequences, file_path
            )

            sequence_data_module = SequenceDataModule(
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                val_sequences=val_sequences,
                batch_size=self.batch_size,
            )
            # Setup the data module to initialize datasets
            sequence_data_module.setup()

            # Create data loaders
            train_loader, val_loader, test_loader = (
                sequence_data_module.train_dataloader(),
                sequence_data_module.val_dataloader(),
                sequence_data_module.test_dataloader(),
            )

            return (
                sequence_length,
                len(FEATURE_COLUMNS),
                train_loader,
                val_loader,
                test_loader,
            )

        return sequences, sequence_length, len(FEATURE_COLUMNS), None, None, None
