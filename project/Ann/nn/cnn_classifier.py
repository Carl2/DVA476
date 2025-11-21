#!/usr/bin/env python
"""
SimpleCNN: A 3-block convolutional neural network for COIL-100 object classification.

Architecture:
- 3 Convolutional blocks with increasing channels (64 → 128 → 256)
- BatchNorm + ReLU activation + MaxPooling in each block
- 2 Fully connected layers with dropout
- Designed for 128x128 RGB input images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pymonad.maybe import Just, Maybe
from typing import Dict, Any



class SimpleCNN(nn.Module):
    """
    Simple CNN with 3 convolutional blocks for image classification.

    Input: (batch_size, 3, 128, 128) RGB images
    Output: (batch_size, num_classes) logits
    """

    def __init__(self, num_classes: int = 100, input_channels: int = 3, dropout_rate: float = 0.5):
        """
        Initialize SimpleCNN.

        Args:
            num_classes: Number of output classes (default: 100 for COIL-100)
            input_channels: Number of input channels (default: 3 for RGB)
            dropout_rate: Dropout probability (default: 0.5)
        """
        super(SimpleCNN, self).__init__()

        # Convolutional Block 1: Extract low-level features
        # 128x128 -> 64x64
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2: Extract mid-level features
        # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3: Extract high-level features
        # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after conv blocks
        # After 3 pooling layers: 128 -> 64 -> 32 -> 16
        # Feature maps: 256 channels × 16 × 16 = 65,536
        self.flatten_size = 256 * 16 * 16

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # (batch, 64, 64, 64)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, 128, 32, 32)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # (batch, 256, 16, 16)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 65536)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (use with CrossEntropyLoss)

        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_model(setting_fn: callable) -> callable:
    def do_create_cnn_model(config:dict) -> Maybe:
        num_classes = setting_fn('num_classes')
        input_channels = setting_fn('input_channels')
        dropout_rate = setting_fn('dropout_rate')
        device = setting_fn('device')

        if (num_classes.is_nothing() or
            input_channels.is_nothing() or
            dropout_rate.is_nothing() or
            device.is_nothing()):

            error_msg = ""
            if num_classes.is_nothing():
                error_msg += f"num_classes setting missing. "
            if input_channels.is_nothing():
                error_msg += f"input_channels setting missing. "
            if dropout_rate.is_nothing():
                error_msg += f"dropout_rate setting missing. "
            if device.is_nothing():
                error_msg += f"Device setting missing. "
            return Maybe(value=f"Failed to get all model configuration settings: {error_msg}", monoid=False)


        model = SimpleCNN(
            num_classes=num_classes.value,
            input_channels=input_channels.value,
            dropout_rate=dropout_rate.value
        )
        # Move to device if specified
        if device.value == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            print(f"✓ Model moved to GPU")
        else:
            print(f"✓ Model on CPU")


        num_params = model.get_num_parameters()
        print(f"✓ SimpleCNN created with {num_params:,} parameters")

        config['model'] = model
        return Just(config)
    return do_create_cnn_model



def save_cnn_model(model_path: Path) -> callable:
    """
    Create a function to save CNN model and training info.

    Args:
        model_path: Path where to save the model

    Returns:
        Function that saves model state and returns Maybe
    """
    def do_save_model(model_data: tuple) -> Maybe:
        try:
            history, model_state = model_data

            torch.save({
                'model_state_dict': model_state,
                'history': history,
                'model_type': 'SimpleCNN'
            }, model_path)

            print(f"✓ CNN model saved to {model_path}")
            return Just(history)

        except Exception as e:
            return Maybe(value=f"Failed to save CNN model: {e}", monoid=False)

    return do_save_model


def load_cnn_model(model_path: Path) -> callable:
    """
    Create a function to load CNN model for inference.

    Args:
        model_path: Path to saved model

    Returns:
        Function that loads model and runs inference
    """
    def do_load_model(test_loader) -> Maybe:
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Create model (you'll need to pass config or hardcode values)
            model = SimpleCNN(num_classes=100, input_channels=3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"✓ CNN model loaded from {model_path}")

            # Run inference
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            all_predictions = []
            all_labels = []
            all_confidences = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)

                    outputs = model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    max_probs, predicted = torch.max(probabilities, 1)

                    all_predictions.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.tolist())
                    all_confidences.extend(max_probs.cpu().tolist())

            # Calculate accuracy
            correct = sum(p == l for p, l in zip(all_predictions, all_labels))
            accuracy = correct / len(all_labels)
            avg_confidence = sum(all_confidences) / len(all_confidences)

            return Just({
                "labels": all_labels,
                "predicted": all_predictions,
                "accuracy": accuracy,
                "confidence": avg_confidence
            })

        except Exception as e:
            return Maybe(value=f"Failed to load/run CNN model: {e}", monoid=False)

    return do_load_model


# I need a train function, that takes a model
def train_cnn_model(model: nn.Module, **kwargs) -> callable:
    """
    Train CNN using raw images (no manual feature extraction needed).
    """
    def do_train_model(train_loader, val_loader) -> Maybe:
        criterion = kwargs.get('loss', nn.CrossEntropyLoss())
        optimizer = kwargs.get('optimizer')
        epochs = kwargs.get("epochs", 50)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "avg_confidence": []
        }

        early_stopping = EarlyStopping(
            patience=kwargs.get('patience', 10),
            min_delta=kwargs.get('min_delta', 0.001)
        )

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_confidences = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)
                    max_probs, predicted = torch.max(probabilities, 1)

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_confidences.extend(max_probs.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            avg_confidence = sum(all_confidences) / len(all_confidences)

            # Record history
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["avg_confidence"].append(avg_confidence)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
                print(f'  Avg Confidence: {avg_confidence*100:.2f}%')

            if early_stopping(val_loss, model):
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        early_stopping.load_best_model(model)
        return Just((history, early_stopping.best_model_state))

    return do_train_model
