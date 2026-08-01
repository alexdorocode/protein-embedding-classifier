from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder


class TorchTrainingWrapper:
    def __init__(
        self,
        input_size: int | None = None,
        output_size: int | None = None,
        hidden_layers_mode: str = "quadratic_increase",
        num_hidden_layers: int = 2,
        custom_hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.1,
        activation_function: str = "ReLU",
        use_batch_norm: bool = False,
        output_activation: str | None = None,
        initialization: str | None = None,
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        criterion_name: str = "CrossEntropyLoss",
        batch_size: int = 64,
        optimizer_group: str | None = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_mode = hidden_layers_mode
        self.num_hidden_layers = num_hidden_layers
        self.custom_hidden_layers = custom_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.use_batch_norm = use_batch_norm
        self.output_activation = output_activation
        self.initialization = initialization

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.criterion_name = criterion_name
        self.batch_size = batch_size
        self.optimizer_group = optimizer_group

        self.logger = logging.getLogger(self.__class__.__name__)
        self.label_encoder = LabelEncoder()
        self.classes_: np.ndarray | None = None

        self.model = None
        self.device = None
        self._torch = None
        self._nn = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        torch, nn = self._get_torch_modules()
        self._torch = torch
        self._nn = nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        y_train_array = np.asarray(y_train)
        y_val_array = np.asarray(y_val)
        is_multilabel_targets = y_train_array.ndim == 2

        if is_multilabel_targets:
            y_train_enc = y_train_array.astype(np.float32)
            y_val_enc = y_val_array.astype(np.float32)
            self.classes_ = np.arange(y_train_enc.shape[1])
            if self.criterion_name != "BCEWithLogitsLoss":
                raise ValueError(
                    "Multilabel targets require BCEWithLogitsLoss, "
                    f"got criterion_name={self.criterion_name}"
                )
        else:
            y_train_enc = self.label_encoder.fit_transform(y_train_array)
            y_val_enc = self.label_encoder.transform(y_val_array)
            self.classes_ = self.label_encoder.classes_

        inferred_input_size = self.input_size if self.input_size is not None else int(X_train.shape[1])
        inferred_output_size = self.output_size if self.output_size is not None else int(len(self.classes_))

        try:
            from protein_embedding_classifier.classifiers.mlp_protein_classifier import MLPProteinClassifier
        except ImportError as exc:
            raise ImportError(
                "MLP model requested but MLPProteinClassifier dependencies are not available. "
                "Install torch with: poetry add torch"
            ) from exc

        self.model = MLPProteinClassifier(
            input_size=inferred_input_size,
            output_size=inferred_output_size,
            num_hidden_layers=int(self.num_hidden_layers),
            dropout_rate=float(self.dropout_rate),
            hidden_layers_mode=self.hidden_layers_mode,
            custom_hidden_layers=self.custom_hidden_layers,
            activation_function=self.activation_function,
            use_batch_norm=bool(self.use_batch_norm),
            output_activation=self.output_activation,
            initialization=self.initialization,
        ).to(self.device)

        optimizer = self._build_optimizer(self.model)
        criterion = self._build_criterion(num_classes=inferred_output_size)

        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        if is_multilabel_targets:
            y_train_tensor = torch.tensor(y_train_enc, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_enc, dtype=torch.float32)
        else:
            y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val_enc, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        configured_batch_size = int(self.batch_size)
        if self.use_batch_norm and configured_batch_size < 2:
            configured_batch_size = 2

        drop_last = (
            bool(self.use_batch_norm)
            and len(dataset) > 1
            and configured_batch_size > 1
            and (len(dataset) % configured_batch_size == 1)
        )
        try:
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=configured_batch_size,
                shuffle=True,
                drop_last=drop_last,
            )
        except TypeError as exc:
            if "drop_last" not in str(exc):
                raise
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=configured_batch_size,
                shuffle=True,
            )

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(int(self.num_epochs)):
            self.model.train()
            train_loss_sum = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self._compute_loss(criterion, logits, y_batch, inferred_output_size)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())

            val_loss = self._evaluate_val_loss(criterion, x_val_tensor, y_val_tensor, inferred_output_size)

            self.logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                train_loss_sum / max(1, len(train_loader)),
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= int(self.early_stopping_patience):
                self.logger.info("Early stopping triggered at epoch=%d", epoch + 1)
                break

        self.model.load_state_dict(best_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self._torch is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        torch = self._torch
        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(x_tensor)

            if logits.ndim == 1:
                logits = logits.unsqueeze(1)

            if logits.shape[1] == 1:
                positive_prob = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                negative_prob = 1.0 - positive_prob
                probs = np.hstack([negative_prob, positive_prob])
                return probs.astype(np.float32)

            if self.criterion_name == "BCEWithLogitsLoss":
                return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            return probs

    def _build_optimizer(self, model):
        torch, _ = self._get_torch_modules()
        optimizer_name = str(self.optimizer_name)
        learning_rate = float(self.learning_rate)

        if optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_name == "RMSprop":
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        if optimizer_name == "SGD":
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        if optimizer_name == "Adagrad":
            return torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    def _build_criterion(self, num_classes: int):
        _, nn = self._get_torch_modules()
        if self.criterion_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        if self.criterion_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()

        raise ValueError(f"Unsupported criterion_name: {self.criterion_name} with num_classes={num_classes}")

    def _compute_loss(self, criterion, logits, y_batch, num_classes: int):
        torch, _ = self._get_torch_modules()
        if self.criterion_name == "CrossEntropyLoss":
            return criterion(logits, y_batch)

        if self.criterion_name == "BCEWithLogitsLoss":
            if y_batch.ndim == 2:
                return criterion(logits, y_batch.float())
            one_hot = torch.nn.functional.one_hot(y_batch, num_classes=num_classes).float()
            return criterion(logits, one_hot)

        raise ValueError(f"Unsupported criterion_name: {self.criterion_name}")

    def _evaluate_val_loss(self, criterion, x_val_tensor, y_val_tensor, num_classes: int) -> float:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.eval()
        with self._torch.no_grad():
            logits = self.model(x_val_tensor.to(self.device))
            y_target = y_val_tensor.to(self.device)
            loss = self._compute_loss(criterion, logits, y_target, num_classes)
        return float(loss.item())

    @staticmethod
    def _get_torch_modules():
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MLP model requested but torch is not installed. "
                "Install it with: poetry add torch"
            ) from exc
        return torch, nn
