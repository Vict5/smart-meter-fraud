# pipeline.py
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch

from .models.config import Config
from .models.trainers import compute_minority_weighted_recall
from .training_orchestrator import TrainingOrchestrator
from .binary_classifiers import BinaryClassifier
from .evaluation import ModelEvaluator
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split
import xgboost as xgb


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class Pipeline:
    """Main pipeline for energy fraud detection"""

    def __init__(self, config):
        self.config = config
        self.training_orchestrator = TrainingOrchestrator(config)
        self.binary_classifier = BinaryClassifier(
            self.training_orchestrator.ae_models_dir,
            config.EMBEDDING_DIM,
            config.DATASET_PATH,
            config.ONLY_REGULAR,
        )
        self.evaluator = ModelEvaluator(config)

    def set_seed(self):
        """Set seed for reproducibility without verbose output"""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run_pipeline(self, mode="XGBoost"):
        """Runs the complete pipeline"""

        # 1. Train autoencoders if necessary
        ae_dir = self.training_orchestrator.ae_models_dir
        if not ae_dir.exists() or len(os.listdir(ae_dir)) != 5:
            self.training_orchestrator.train_all_ae()

        # 2. Generate embeddings if necessary
        emb_dir = self.training_orchestrator.embedding_path
        if not emb_dir.exists() or len(os.listdir(emb_dir)) != 5:
            print("AE ALREADY TRAINED, creating embeddings...\n")
            self.training_orchestrator.generate_all_embeddings()

        # 3. Combine embeddings if necessary
        comb_dir = self.training_orchestrator.combined_embeddings_path
        if not comb_dir.exists() or len(os.listdir(comb_dir)) != 1:
            print("EMBEDDINGS ALREADY CREATED, combining...\n")
            combined_embeddings = self.training_orchestrator.combine_embeddings()
        else:
            out_dir = os.path.join(comb_dir, "combined_embeddings.npy")
            combined_embeddings = np.load(out_dir)
            print("combined embeddings already present")

        # 4. Run the requested mode
        if mode == "Binary_Classifier" and self.config.ONLY_REGULAR:
            self._run_binary_classifier(combined_embeddings)
        elif mode == "DNN_Classifier":
            self._run_dnn_classifier(combined_embeddings)
        elif mode == "XGBoost":
            self._run_xgboost_classifier(combined_embeddings)
        elif mode == "RandomForest":
            self._run_randomforest_classifier(combined_embeddings)

    def _run_binary_classifier(self, combined_embeddings):
        """Runs the binary classifier"""
        labels_path = os.path.join(self.config.DATASET_PATH, "LABELS.csv")
        labels = pd.read_csv(labels_path, sep="\t", encoding="utf-16")
        labels = labels.sort_values("Supply_ID").reset_index(drop=True)
        y_true = labels["CLUSTER"].to_numpy()
        y_true = [0 if label == 2 else 1 for label in y_true]
        y_pred = self.binary_classifier.binary_classifier_combined_embeddings(
            combined_embeddings, y_true
        )

        self.evaluator.evaluate_binary_classifier(y_true, y_pred, "Binary Classifier")

    def _run_dnn_classifier(self, combined_embeddings):
        """Runs the DNN classifier"""
        X, y = self._prepare_classification_data(combined_embeddings)
        hard_voting_pred, soft_voting_pred, y_test = (
            self.evaluator.train_and_evaluate_dnn(X, y)
        )

        print(classification_report(y_test, hard_voting_pred))
        print(classification_report(y_test, soft_voting_pred))
        print(
            "Weighted-recall hv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, hard_voting_pred)
            )
        )
        print(
            "Weighted-recall sv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, soft_voting_pred)
            )
        )

        cmhv = confusion_matrix(y_test, hard_voting_pred)
        cmsf = confusion_matrix(y_test, soft_voting_pred)

        fig, ax = plt.subplots(1, 2)

        disphv = ConfusionMatrixDisplay(confusion_matrix=cmhv, display_labels=[0, 1, 2])
        dispsv = ConfusionMatrixDisplay(confusion_matrix=cmsf, display_labels=[0, 1, 2])

        fig.suptitle("Confusion Matrix For DNN Ensambles Classifiers")

        ax[0].set_title("HARD VOTING")
        ax[1].set_title("SOFT VOTING")

        disphv.plot(ax=ax[0], colorbar=False, cmap="Greens")
        dispsv.plot(ax=ax[1], colorbar=False, cmap="Greens")
        plt.show()

        # Calculate feature contributions with SHAP for DNN
        self._calculate_dataset_contributions_shap(X, y)

        print("DNN Classification completed")

    def _run_xgboost_classifier(self, combined_embeddings):
        """Runs the XGBoost classifier"""
        X, y = self._prepare_classification_data(combined_embeddings)
        hard_voting_pred, soft_voting_pred, y_test = (
            self.evaluator.train_and_evaluate_xgboost(
                X, y, self.config.MAX_DEPTH, self.config.LEARNING_RATE_XG
            )
        )

        print(classification_report(y_test, hard_voting_pred))
        print(classification_report(y_test, soft_voting_pred))
        print(
            "Weighted-recall hv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, hard_voting_pred)
            )
        )
        print(
            "Weighted-recall sv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, soft_voting_pred)
            )
        )

        print(confusion_matrix(y_test, hard_voting_pred))
        print(confusion_matrix(y_test, soft_voting_pred))

        cmhv = confusion_matrix(y_test, hard_voting_pred)
        cmsf = confusion_matrix(y_test, soft_voting_pred)

        fig, ax = plt.subplots(1, 2)

        disphv = ConfusionMatrixDisplay(confusion_matrix=cmhv, display_labels=[0, 1, 2])
        dispsv = ConfusionMatrixDisplay(confusion_matrix=cmsf, display_labels=[0, 1, 2])

        fig.suptitle("Confusion Matrix For XGBOOST Ensambles Classifiers")

        ax[0].set_title("HARD VOTING")
        ax[1].set_title("SOFT VOTING")

        disphv.plot(ax=ax[0], colorbar=False, cmap="Greens")
        dispsv.plot(ax=ax[1], colorbar=False, cmap="Greens")
        plt.show()

        # Calculate feature contributions with SHAP for XGBoost
        self._calculate_dataset_contributions_shap_xgboost(X, y)

        print("XGBoost Classification completed")

    def _run_randomforest_classifier(self, combined_embeddings):
        """Runs the Random Forest classifier"""
        X, y = self._prepare_classification_data(combined_embeddings)
        hard_voting_pred, soft_voting_pred, y_test = (
            self.evaluator.train_and_evaluate_randomforest(X, y)
        )

        print(classification_report(y_test, hard_voting_pred))
        print(classification_report(y_test, soft_voting_pred))
        print(
            "Weighted-recall hv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, hard_voting_pred)
            )
        )
        print(
            "Weighted-recall sv= {:.4f}".format(
                compute_minority_weighted_recall(y_test, soft_voting_pred)
            )
        )
        print(confusion_matrix(y_test, hard_voting_pred))
        print(confusion_matrix(y_test, soft_voting_pred))

        cmhv = confusion_matrix(y_test, hard_voting_pred)
        cmsf = confusion_matrix(y_test, soft_voting_pred)

        fig, ax = plt.subplots(1, 2)

        disphv = ConfusionMatrixDisplay(confusion_matrix=cmhv, display_labels=[0, 1, 2])
        dispsv = ConfusionMatrixDisplay(confusion_matrix=cmsf, display_labels=[0, 1, 2])

        fig.suptitle("Confusion Matrix For RANDOM FOREST Ensambles Classifiers")

        ax[0].set_title("HARD VOTING")
        ax[1].set_title("SOFT VOTING")

        disphv.plot(ax=ax[0], colorbar=False, cmap="Greens")
        dispsv.plot(ax=ax[1], colorbar=False, cmap="Greens")
        plt.show()

        # Calculate feature contributions with SHAP for Random Forest
        self._calculate_dataset_contributions_shap_randomforest(X, y)

        print("Random Forest Classification completed")

    def _prepare_classification_data(self, combined_embeddings):
        """Prepares data for classification"""
        X = combined_embeddings
        labels_path = os.path.join(self.config.DATASET_PATH, "LABELS.csv")
        labels = pd.read_csv(labels_path, sep="\t", encoding="utf-16")
        # Sort by Supply_ID to align row order with combined_embeddings (built via Combiner
        # which iterates over np.unique(supply_ids), i.e. sorted order).
        labels = labels.sort_values("Supply_ID").reset_index(drop=True)
        y = labels["CLUSTER"].to_numpy()
        return X, y

    def _calculate_dataset_contributions_shap(self, X, y):
        """Calculates feature contributions for each dataset using SHAP with DNN"""

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            stratify=y,
        )

        # Train a simple DNN for SHAP
        from sklearn.neural_network import MLPClassifier

        dnn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=self.config.SEED,
        )
        dnn_model.fit(X_train, y_train)

        # Use generic Explainer for DNN (slower but accurate)
        # Take a subset for background
        background_size = min(50, len(X_train))
        background = X_train[:background_size]

        explainer = shap.Explainer(dnn_model.predict_proba, background)

        # Calculate SHAP values for a subset of test data (for speed)
        sample_size = min(50, len(X_test))  # Reduced for speed with DNN
        X_sample = X_test[:sample_size]
        shap_values = explainer(X_sample)

        # For multiclass classification, take the mean of absolute values across all classes
        if len(shap_values.shape) == 3:
            # Multiclass: [samples, features, classes]
            mean_shap_values = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            # Binary: [samples, features]
            mean_shap_values = np.abs(shap_values.values).mean(axis=0)

        # Define dataset names in concatenation order
        dataset_names = [
            "ANAGRAFICA",
            "CONSUMI",
            "INTERRUZIONI",
            "LAVORI",
            "PAROLE_DI_STATO",
        ]
        # Each dataset contributes (embedding_dim + 1) features: embedding + reconstruction error
        features_per_dataset = self.config.EMBEDDING_DIM + 1

        # Calculate contributions per dataset (mean of SHAP values)
        dataset_contributions = {}
        total_importance = 0

        for i, dataset_name in enumerate(dataset_names):
            start_idx = i * features_per_dataset
            end_idx = (i + 1) * features_per_dataset
            dataset_importance = np.mean(mean_shap_values[start_idx:end_idx])
            dataset_contributions[dataset_name] = dataset_importance
            total_importance += dataset_importance

        # Normalize contributions to percentage
        print("\n" + "=" * 50)
        print("DATASET SHAP CONTRIBUTIONS ANALYSIS (DNN)")
        print("=" * 50)

        for dataset_name, importance in dataset_contributions.items():
            percentage = (importance / total_importance) * 100
            print(f"• {dataset_name} embeddings: {percentage:.1f}% contribution")

        print("=" * 50 + "\n")

        return dataset_contributions

    def _calculate_dataset_contributions_shap_xgboost(self, X, y):
        """Calculates feature contributions for each dataset using SHAP with XGBoost"""

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            stratify=y,
        )

        # Train an XGBoost model
        xgb_model = xgb.XGBClassifier(
            max_depth=self.config.MAX_DEPTH,
            learning_rate=self.config.LEARNING_RATE_XG,
            random_state=self.config.SEED,
            n_estimators=100,
        )
        xgb_model.fit(X_train, y_train)

        # Use SHAP TreeExplainer for XGBoost (very efficient)
        explainer = shap.TreeExplainer(xgb_model)

        # Calculate SHAP values for a subset of test data (for speed)
        sample_size = min(100, len(X_test))
        X_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        # For multiclass classification, take the mean of absolute values across all classes
        if isinstance(shap_values, list):
            # Multiclass: shap_values is a list
            mean_shap_values = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            # Binary: shap_values is an array
            mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Define dataset names in concatenation order
        dataset_names = [
            "ANAGRAFICA",
            "CONSUMI",
            "INTERRUZIONI",
            "LAVORI",
            "PAROLE_DI_STATO",
        ]
        features_per_dataset = self.config.EMBEDDING_DIM + 1

        # Calculate contributions per dataset (mean of SHAP values)
        dataset_contributions = {}
        total_importance = 0

        for i, dataset_name in enumerate(dataset_names):
            start_idx = i * features_per_dataset
            end_idx = (i + 1) * features_per_dataset
            dataset_importance = np.mean(mean_shap_values[start_idx:end_idx])
            dataset_contributions[dataset_name] = dataset_importance
            total_importance += dataset_importance

        # Normalize contributions to percentage
        print("\n" + "=" * 50)
        print("DATASET SHAP CONTRIBUTIONS ANALYSIS (XGBoost)")
        print("=" * 50)

        for dataset_name, importance in dataset_contributions.items():
            percentage = (importance / total_importance) * 100
            print(f"• {dataset_name} embeddings: {percentage:.1f}% contribution")

        print("=" * 50 + "\n")

        return dataset_contributions

    def _calculate_dataset_contributions_shap_randomforest(self, X, y):
        """Calculates feature contributions for each dataset using SHAP with Random Forest"""

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            stratify=y,
        )

        # Train a Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=self.config.SEED,
            class_weight="balanced",  # Automatically handles class imbalance
        )
        rf_model.fit(X_train, y_train)

        # Use SHAP TreeExplainer for Random Forest (very efficient)
        explainer = shap.TreeExplainer(rf_model)

        # Calculate SHAP values for a subset of test data (for speed)
        sample_size = min(100, len(X_test))
        X_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        # For multiclass classification, take the mean of absolute values across all classes
        if isinstance(shap_values, list):
            # Multiclass: shap_values is a list
            mean_shap_values = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            # Binary: shap_values is an array
            mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Define dataset names in concatenation order
        dataset_names = [
            "ANAGRAFICA",
            "CONSUMI",
            "INTERRUZIONI",
            "LAVORI",
            "PAROLE_DI_STATO",
        ]
        features_per_dataset = self.config.EMBEDDING_DIM + 1

        # Calculate contributions per dataset (mean of SHAP values)
        dataset_contributions = {}
        total_importance = 0

        for i, dataset_name in enumerate(dataset_names):
            start_idx = i * features_per_dataset
            end_idx = (i + 1) * features_per_dataset
            dataset_importance = np.mean(mean_shap_values[start_idx:end_idx])
            dataset_contributions[dataset_name] = dataset_importance
            total_importance += dataset_importance

        # Normalize contributions to percentage
        print("\n" + "=" * 50)
        print("DATASET SHAP CONTRIBUTIONS ANALYSIS (Random Forest)")
        print("=" * 50)

        for dataset_name, importance in dataset_contributions.items():
            percentage = (importance / total_importance) * 100
            print(f"• {dataset_name} embeddings: {percentage:.1f}% contribution")

        print("=" * 50 + "\n")

        return dataset_contributions


def main(mode=None):
    """Entry point for both CLI and notebook use.

    From the CLI:   python -m src.pipeline --mode XGBoost
    From a notebook / Colab:  main(mode="XGBoost")
    """
    if mode is None:
        parser = argparse.ArgumentParser(description="Run pipeline with mode")
        parser.add_argument(
            "--mode",
            type=str,
            choices=["Binary_Classifier", "DNN_Classifier", "XGBoost", "RandomForest"],
            required=True,
            help="Select the pipeline mode",
        )
        # parse_known_args ignores Jupyter/Colab's extra sys.argv entries
        args, _ = parser.parse_known_args()
        mode = args.mode

    pipeline = Pipeline(Config)
    pipeline.set_seed()
    pipeline.run_pipeline(mode)


if __name__ == "__main__":
    main()
