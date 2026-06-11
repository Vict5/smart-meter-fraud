# data_module.py
import torch
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label, id = self.sequences[idx]
        return dict(
            sequence=torch.FloatTensor(sequence),
            label=torch.tensor(label, dtype=torch.long),
            id=torch.tensor(id, dtype=torch.long),
        )


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, val_sequences, batch_size=8):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_size
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = SequenceDataset(self.train_sequences)
        self.test_dataset = SequenceDataset(self.test_sequences)
        self.val_dataset = SequenceDataset(
            self.val_sequences
        )  # Fix: Create proper dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing for better performance on small datasets
        )

    def val_dataloader(self):
        # Fix: Return DataLoader with proper dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for better performance
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for better performance
        )
