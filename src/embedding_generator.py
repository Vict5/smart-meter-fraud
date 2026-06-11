# embedding_generator.py
import os
import numpy as np
import torch
from src.models.Autoencoder.models import RecurrentAutoencoder
from src.data_processing import DataProcessor


class EmbeddingGenerator:
    """Manages the creation of embeddings from trained models"""

    def __init__(self, embedding_dim, ae_models_dir, embedding_path):
        self.embedding_dim = embedding_dim
        self.ae_models_dir = ae_models_dir
        self.embedding_path = embedding_path

    def create_embedding(self, ds_name, only_regular, dataset_path):
        print(f"Processing model file: {ds_name}")

        model_path = os.path.join(
            self.ae_models_dir, ds_name[:-4] + "_autoencoder_model.pth"
        )

        # Create a DataProcessor instance
        data_processor = DataProcessor(dataset_path, None, None, None, None)

        sequences, seq_len, n_features, _, _, _ = (
            data_processor.load_and_preprocess_data(
                ds_name, only_regular=only_regular, for_training=False
            )
        )

        model = RecurrentAutoencoder(seq_len, n_features, self.embedding_dim)

        # Load the checkpoint which contains model_state_dict, history, and config
        checkpoint = torch.load(model_path, weights_only=False)

        # Extract only the model state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Prepare data for inference - convert sequences to tensor
        sequence_data = []
        labels = []
        supply_ids = []
        for sequence, label, supply_id in sequences:
            sequence_data.append(sequence)
            labels.append(label)
            supply_ids.append(supply_id)

        # Convert to tensor
        input_tensor = torch.FloatTensor(np.array(sequence_data))
        print(f"Input tensor shape: {input_tensor.shape}")

        # Get embeddings using the encoder only
        with torch.no_grad():
            embeddings = model.encoder(input_tensor)
            reconstructed = model(input_tensor)
            reconstruction_error = torch.mean(
                torch.abs(reconstructed - input_tensor), dim=(1, 2)
            )

        reconstruction_error = reconstruction_error.cpu().numpy()
        supply_ids = np.array(supply_ids)
        # concatenate embeddings with supply_ids
        embeddings_with_ids = np.concatenate(
            (
                supply_ids[:, np.newaxis],
                embeddings.numpy(),
                reconstruction_error[:, np.newaxis],
            ),
            axis=1,
        )
        print(f"Embeddings with ID and Rec error shape: {embeddings_with_ids.shape}")
        print(embeddings_with_ids[4, :])

        os.makedirs(self.embedding_path, exist_ok=True)
        np.save(
            f"{self.embedding_path}/{ds_name[:-4]}_embeddings.npy", embeddings_with_ids
        )
        print(f"Saved embeddings to {self.embedding_path}{ds_name[:-4]}_embeddings.npy")
