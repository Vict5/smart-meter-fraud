# training_orchestrator.py
import os
import pathlib
from src.models.trainers import TrainAutoEncoder
from src.data_processing import DataProcessor
from src.embedding_generator import EmbeddingGenerator
from src.models.combine import Combiner
import pandas as pd
import numpy as np


class TrainingOrchestrator:
    """Coordinates the training of all models"""

    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(
            config.DATASET_PATH,
            config.BATCH_SIZE,
            config.TEST_SIZE,
            config.VAL_SIZE,
            config.SEED,
        )

        # Setup directories
        self.ae_models_dir = (
            pathlib.Path("models/Autoencoder/only_regular")
            if config.ONLY_REGULAR
            else pathlib.Path("models/Autoencoder/all_clusters")
        )
        self.embedding_path = (
            pathlib.Path("data/embeddings/only_regular")
            if config.ONLY_REGULAR
            else pathlib.Path("data/embeddings/all_clusters")
        )
        self.combined_embeddings_path = (
            pathlib.Path("data/combined_embeddings/only_regular")
            if config.ONLY_REGULAR
            else pathlib.Path("data/combined_embeddings/all_clusters")
        )
        self.reports_dir_ae = (
            pathlib.Path("reports/figures/ae_training_loss/only_regular")
            if config.ONLY_REGULAR
            else pathlib.Path("reports/figures/ae_training_loss/all_clusters")
        )

    def train_all_ae(self):
        """Trains all autoencoders"""
        for file in self.config.DATASET_PATH.glob("*.csv"):
            ds_name = file.name
            if ds_name == "LABELS.csv":
                continue

            print(f"TRAIN {ds_name}")

            seq_len, n_features, train_loader, val_loader, test_loader = (
                self.data_processor.load_and_preprocess_data(
                    ds_name, self.config.ONLY_REGULAR
                )
            )

            ae_trainer = TrainAutoEncoder(
                seq_len, n_features, ds_name, train_loader, val_loader, test_loader
            )

            ae_history, train_loss_complete = ae_trainer.train_autoencoder(
                self.config.EMBEDDING_DIM,
                self.ae_models_dir,
                self.config.LEARNING_RATE_AE,
                self.config.N_EPOCHS_AE,
            )
            ae_trainer.plot_training_results(ds_name, ae_history, self.reports_dir_ae)

            print(f"TESTING {ds_name}")

            ae_trainer.validate_autoencoder(
                seq_len=seq_len,
                n_features=n_features,
                ae_models_dir=self.ae_models_dir,
                embedding_dim=self.config.EMBEDDING_DIM,
            )
            if self.config.ONLY_REGULAR:
                print("training with only regualr samples")

    def generate_all_embeddings(self):
        """Generates all embeddings"""
        embedding_generator = EmbeddingGenerator(
            self.config.EMBEDDING_DIM, self.ae_models_dir, self.embedding_path
        )

        for file in self.config.DATASET_PATH.glob("*.csv"):
            ds_name = file.name
            if ds_name == "LABELS.csv":
                continue
            # Always generate embeddings for ALL customers regardless of ONLY_REGULAR.
            # ONLY_REGULAR controls what the AE was trained on; inference must cover
            # fraud/anomaly samples too so downstream classifiers can see them.
            embedding_generator.create_embedding(
                ds_name, only_regular=False, dataset_path=self.config.DATASET_PATH
            )

    def combine_embeddings(self):
        """Combines all embeddings"""
        embeddings_dim = 17

        # Determine the number of IDs based on the mode
        labels_path = os.path.join(self.config.DATASET_PATH, "LABELS.csv")
        labels = pd.read_csv(labels_path, sep="\t", encoding="utf-16")
        num_ids = len(labels)

        combiner = Combiner(
            self.embedding_path, num_ids=num_ids, embedding_dim=embeddings_dim
        )
        combined_embeddings = combiner.concatenate_embeddings()
        print(f"Combined embeddings shape: {combined_embeddings.shape}")

        os.makedirs(self.combined_embeddings_path, exist_ok=True)
        out_dir = os.path.join(self.combined_embeddings_path, "combined_embeddings.npy")
        np.save(out_dir, combined_embeddings)

        return combined_embeddings
