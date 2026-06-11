# binary_classifiers.py
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import pandas as pd
from src.models.Autoencoder.models import RecurrentAutoencoder
import matplotlib.pyplot as plt


class BinaryClassifier:
    """Manages binary classifiers based on reconstruction error threshold"""

    def __init__(
        self, ae_models_dir, embedding_dim, dataset_path, only_regular, alpha=0.5
    ):
        self.ae_models_dir = ae_models_dir
        self.embedding_dim = embedding_dim
        self.dataset_path = dataset_path
        self.only_regular = only_regular
        self.alpha = alpha

    def binary_classifier_single_dataset(
        self, train_loss, seq_len, n_features, ds_name, loader, y_true
    ):
        y_pred = []
        treshold = np.mean(train_loss) + np.std(train_loss)
        print("threshold: ", treshold)

        model = RecurrentAutoencoder(seq_len, n_features, self.embedding_dim)
        loss_fn = torch.nn.L1Loss(reduction="none")

        model_path = f"{self.ae_models_dir}/{ds_name[:-4]}_autoencoder_model.pth"
        checkpoint = torch.load(model_path, weights_only=False)
        # Extract only the model state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        y_scores = []
        with torch.no_grad():
            for batch in loader:
                sequences = batch["sequence"]
                output = model(sequences)
                loss_per_sample = loss_fn(output, sequences)
                loss_per_sample = loss_per_sample.view(
                    loss_per_sample.size(0), -1
                ).mean(dim=1)
                for loss_value in loss_per_sample:
                    pred = 0 if loss_value.item() < treshold else 1
                    y_pred.append(pred)
                    y_scores.append(loss_value)

        return y_pred

    def binary_classifier_combined_embeddings(self, combined_embeddings, y_true):
        y_pred = []
        y_score = []
        # Calibrate the threshold using only regular customers (when ONLY_REGULAR=True),
        # because the AE was trained on normal data and that is the reference distribution.
        # combined_embeddings rows are ordered by sorted Supply_ID (see Combiner), so we
        # must sort LABELS.csv the same way before building the boolean mask.
        if self.only_regular:
            labels_path = os.path.join(self.dataset_path, "LABELS.csv")
            labels = pd.read_csv(labels_path, encoding="utf-16", sep="\t")
            labels_sorted = labels.sort_values("Supply_ID").reset_index(drop=True)
            regular_mask = labels_sorted["CLUSTER"].values == 2
            filtered_embeddings = combined_embeddings[regular_mask]
        else:
            filtered_embeddings = combined_embeddings

        # Extract all rec_errors (every 17th starting from 16)
        all_rec_errors = filtered_embeddings[:, 16::17].flatten()

        # Remove zeros
        non_zero_rec_errors = all_rec_errors[all_rec_errors != 0]

        # Calculate threshold consistently
        treshold = np.mean(non_zero_rec_errors) + self.alpha * np.std(
            non_zero_rec_errors
        )

        print("Threshold:", treshold)

        # Calculate prediction for each embedding (use original embeddings for prediction)
        for embedding in combined_embeddings:
            rec_errors = embedding[16::17]
            non_zero_errors = rec_errors[rec_errors != 0]

            if non_zero_errors.size > 0:
                rec_error_mean = np.mean(non_zero_errors)
            else:
                rec_error_mean = 0

            pred = 0 if rec_error_mean < treshold else 1
            y_pred.append(pred)
            y_score.append(rec_error_mean)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        return y_pred
