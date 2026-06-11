# models.py
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import pathlib
import os

from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

import xgboost as xgb


from torch.utils.data import TensorDataset, DataLoader

from src.models.config import Config
from src.models.Autoencoder.models import RecurrentAutoencoder
from src.models.DNN.models import ClassifierMLP


def compute_minority_weighted_recall(y_true, y_pred):
    """
    Calculates weighted recall giving higher weight to minority class
    """
    # Count occurrences of each class
    class_counts = Counter(y_true)
    total_samples = len(y_true)

    # Calculate weights: minority classes have higher weight
    class_weights = {}
    for class_label, count in class_counts.items():
        # Weight inversely proportional to class frequency
        class_weights[class_label] = total_samples / (len(class_counts) * count)

    # Calculate recall for each class
    unique_labels = list(class_counts.keys())
    recall_per_class = recall_score(
        y_true, y_pred, labels=unique_labels, average=None, zero_division=0
    )

    # Calculate weighted average
    weighted_recall = 0
    total_weight = 0
    for i, class_label in enumerate(unique_labels):
        weight = class_weights[class_label]
        weighted_recall += recall_per_class[i] * weight
        total_weight += weight

    return weighted_recall / total_weight if total_weight > 0 else 0


class TrainAutoEncoder:
    def __init__(
        self, seq_len, n_features, ds_name, train_loader, val_loader, test_loader
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.ds_name = ds_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_autoencoder(self, embedding_dim, ae_models_dir, lr, epochs):
        ae_history = dict(train_mae=[], val_mae=[])
        train_loss_complete = []  # save loss for every sequence

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = RecurrentAutoencoder(self.seq_len, self.n_features, embedding_dim)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Use mean reduction instead of sum for better interpretability
        train_loss_fn = torch.nn.L1Loss(reduction="none")
        val_loss_fn = torch.nn.L1Loss(reduction="mean")

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float("inf")

        print("Epoch | Train MAE | Val MAE")
        for epoch in range(1, epochs + 1):
            # train mode
            model.train()
            train_maes = []

            for batch in self.train_loader:
                # take only the sequences
                sequences = batch["sequence"]

                optimizer.zero_grad()
                output = model(sequences)
                loss_per_sample = train_loss_fn(
                    output, sequences
                )  # the target is the original sequence
                loss_per_sample = loss_per_sample.view(
                    loss_per_sample.size(0), -1
                ).mean(dim=1)

                loss = loss_per_sample.mean()
                loss.backward()
                optimizer.step()

                train_loss_complete.extend(loss_per_sample.detach().cpu().tolist())

                # save loss
                train_maes.append(loss.item())

            # validation mode
            model.eval()
            val_maes = []

            with torch.no_grad():
                for batch in self.val_loader:
                    sequences = batch["sequence"]

                    output = model(sequences)
                    loss = val_loss_fn(
                        output, sequences
                    )  # the target is the original sequence
                    val_maes.append(loss.item())

            # Calculate avg losses
            train_mae = np.mean(train_maes)
            val_mae = np.mean(val_maes)

            ae_history["train_mae"].append(train_mae)
            ae_history["val_mae"].append(val_mae)

            # save best model
            if val_mae < best_loss:
                best_loss = val_mae
                best_model_wts = copy.deepcopy(model.state_dict())

            if epoch % 50 == 0 or epoch == epochs:
                print(f"{epoch:5d} | {train_mae:10.4f} | {val_mae:8.4f} ")

        model.load_state_dict(best_model_wts)
        pathlib.Path(ae_models_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": ae_history,
                "config": {
                    "seq_len": self.seq_len,
                    "n_features": self.n_features,
                    "embedding_dim": embedding_dim,
                },
            },
            f"{ae_models_dir}/{self.ds_name[:-4]}_autoencoder_model.pth",
        )
        print(f"\nBest validation loss: {best_loss:.4f}")
        return ae_history, train_loss_complete

    def validate_autoencoder(self, seq_len, n_features, ae_models_dir, embedding_dim):
        model = RecurrentAutoencoder(seq_len, n_features, embedding_dim)
        loss_fn = torch.nn.L1Loss(reduction="mean")

        model_path = f"{ae_models_dir}/{self.ds_name[:-4]}_autoencoder_model.pth"
        checkpoint = torch.load(model_path, weights_only=False)
        # Extract only the model state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_maes = []
        with torch.no_grad():
            for batch in self.test_loader:
                sequences = batch["sequence"]
                print(sequences.shape)
                labels = batch["label"]
                output = model(sequences)
                loss = loss_fn(output, sequences)  # the target is the original sequence
                test_maes.append(loss.item())
                print(f"MAE: {loss.item()} | Label {labels}")

    def plot_training_results(self, ds_name, history, reports_dir):
        n_epochs = len(history["train_mae"])
        fig, ax = plt.subplots()
        ax.plot(range(n_epochs), history["train_mae"], label="Train MAE", color="blue")
        ax.plot(range(n_epochs), history["val_mae"], label="Val MAE", color="orange")
        ax.set_title(f"Training and Validation Loss on {ds_name[:-3]}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        pathlib.Path(reports_dir).mkdir(parents=True, exist_ok=True)
        name_file = pathlib.Path(f"{ds_name[:-4]}_training_losses.jpg")
        out_path = os.path.join(reports_dir, name_file)
        plt.savefig(out_path)
        plt.close(fig)


class TrainDNN:
    def __init__(self, X_train, X_val=None, y_train=None, y_val=None):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def train_neural_network(
        self, lr, epochs, dnn_model_dir, batch_size, weight_decay, fold
    ):
        dnn_history = dict(train_loss=[], val_loss=[])

        """Train neural network for one fold"""
        model = ClassifierMLP(input_dim=self.X_train.shape[1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # Move model to device

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(self.y_train), y=self.y_train.numpy()
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", weight=class_weights_tensor
        )

        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training with early stopping
        best_val_loss = float("inf")
        patience = 30
        patience_counter = 0

        train_loss = []
        val_loss = []

        print(
            "Epoch | Train loss | Val loss | Train Rec | Train Acc | Val rec | Vale Acc"
        )
        print(
            "      |            |          | (min-wgt) |           |(min-wgt)|         "
        )
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = []
            train_pred = []
            train_true = []
            train_recall, train_acc = 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                # prediction
                _, predicted = torch.max(outputs, 1)
                train_pred.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())

                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            # Calculate weighted recall favoring minority class
            train_recall = compute_minority_weighted_recall(train_true, train_pred)
            train_acc = accuracy_score(train_true, train_pred)

            # Validation phase
            model.eval()
            val_loss = []
            val_pred = []
            val_true = []
            val_recall, val_acc = 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    val_loss.append(loss.item())

                    _, predicted = torch.max(outputs, 1)
                    val_pred.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

                # Calculate weighted recall favoring minority class
                val_recall = compute_minority_weighted_recall(val_true, val_pred)
                val_acc = accuracy_score(val_true, val_pred)

            avg_train_loss = np.mean(train_loss)
            avg_val_loss = np.mean(val_loss)

            dnn_history["train_loss"].append(avg_train_loss)
            dnn_history["val_loss"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_recall_score = val_recall
                patience_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if epoch % 50 == 0 or epoch == epochs:
                train_recall_str = f"{train_recall:8.3f}"
                val_recall_str = f"{val_recall:8.3f}"
                print(
                    f"{epoch:5d} | {avg_train_loss:10.4f} | {avg_val_loss:8.4f} | {train_recall_str} | {train_acc:8.4f}  | {val_recall_str} | {val_acc:8.4f}"
                )

        model.load_state_dict(best_model_wts)

        # Save best model state for testing
        self.best_model_state = copy.deepcopy(best_model_wts)

        pathlib.Path(dnn_model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": dnn_history,
            },
            f"{dnn_model_dir}/dnn_model_fold_{fold}.pth",
        )
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Best recall : {best_recall_score}")
        return dnn_history, best_recall_score

    def test_dnn(self, X_test, y_test, model_path=None):
        model = ClassifierMLP(input_dim=self.X_train.shape[1])

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            output = model(X_test)
            # Apply softmax
            probabilities = torch.softmax(output, dim=1)
            y_pred = torch.argmax(probabilities, axis=1)
        # print(accuracy_score(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        return y_pred, probabilities


class TrainXGBoost:
    def __init__(self, X_train, X_val=None, y_train=None, y_val=None):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def train_xgboost(self, xgboost_model_dir, max_depth, lr, fold):
        class_counts = np.bincount(self.y_train.numpy())
        unique_classes = np.unique(self.y_train.numpy())

        # For multiclass imbalanced data, we'll use class_weight in sample_weight
        sample_weights = np.ones(len(self.y_train))
        for i, class_label in enumerate(unique_classes):
            class_weight = len(self.y_train) / (len(unique_classes) * class_counts[i])
            sample_weights[self.y_train == class_label] = class_weight

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=100,
            learning_rate=lr,
            subsample=0.8,  # Prevent overfitting
            colsample_bytree=0.8,  # Prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=Config.SEED,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
        )

        # Fit with sample weights for class imbalance
        xgb_model.fit(
            self.X_train.numpy(),
            self.y_train.numpy(),
            sample_weight=sample_weights,
            eval_set=[(self.X_val.numpy(), self.y_val.numpy())],
            verbose=False,
        )

        y_pred_val = xgb_model.predict(self.X_val.numpy())
        recall = compute_minority_weighted_recall(self.y_val, y_pred_val)

        pathlib.Path(xgboost_model_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(xgboost_model_dir, "fold_" + str(fold) + ".json")
        xgb_model.save_model(out_path)

        return xgb_model, recall

    def test_xgboost(self, X_test, xgb_model):
        """Test XGBoost model on test data"""
        y_pred = xgb_model.predict(X_test.numpy())
        y_pred_proba = xgb_model.predict_proba(X_test.numpy())

        return y_pred, y_pred_proba


class TrainRandomForest:
    """Training and testing Random Forest models"""

    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def train_randomforest(self, models_dir, n_estimators=200, max_depth=10, fold=0):
        """Train Random Forest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        # Create directory if it doesn't exist
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)

        # Calculate class weights to handle class imbalance
        classes = np.unique(self.y_train.numpy())
        class_weights = compute_class_weight(
            "balanced", classes=classes, y=self.y_train.numpy()
        )
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

        # Inizializza il modello Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight_dict,
            n_jobs=-1,  # Use all processors
        )

        # Train the model
        print(
            f"Training Random Forest with {n_estimators} trees, max_depth={max_depth}"
        )
        rf_model.fit(self.X_train.numpy(), self.y_train.numpy())

        # Valutazione sul validation set
        y_pred_val = rf_model.predict(self.X_val.numpy())
        recall = compute_minority_weighted_recall(self.y_val.numpy(), y_pred_val)

        print(f"Validation weighted recall: {recall:.4f}")

        # Save the model
        model_path = os.path.join(models_dir, f"fold_{fold}.pkl")
        joblib.dump(rf_model, model_path)
        print(f"Random Forest model saved to {model_path}")

        return rf_model, recall

    def test_randomforest(self, X_test, rf_model):
        """Test Random Forest model on test data"""
        y_pred = rf_model.predict(X_test.numpy())
        y_pred_proba = rf_model.predict_proba(X_test.numpy())

        return y_pred, y_pred_proba
