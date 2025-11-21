#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import ConfusionMatrix, Accuracy
from pymonad.maybe import Maybe, Just
from tqdm import tqdm
from pathlib import Path
import csv
from common.utils import config_helper

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN, self).__init__()

        # Conv Block 1
        # 32 filters and kernel size 3, using padding=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 16 * 64, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv Block 1: 64×64×3 → 32×32×32
        x = self.conv1(x)  # 64×64×32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # 32×32×32

        # Conv Block 2: 32×32×32 → 16×16×64
        x = self.conv2(x)  # 32×32×64
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool2(x)  # 16×16×64

        # Flatten: 16×16×64 → 16384
        x = x.view(x.size(0), -1)  # or x.flatten(1)

        # FC Layers
        x = self.fc1(x)  # 128
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 10

        return x


def log_metrics(configs: dict) -> Maybe:
    """Log metrics to CSV file."""
    try:
        logging_config = configs.get("logging", {})
        csv_path = logging_config.get("metrics_csv", "metrics.csv")

        # Create directory if needed
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to write header
        file_exists = Path(csv_path).exists()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'train_acc',
                                'val_loss', 'val_acc'])

            # Write metrics
            writer.writerow([
                configs['current_epoch'],
                f"{configs['epoch_loss']:.4f}",
                f"{configs['epoch_acc']:.2f}",
                f"{configs['val_loss']:.4f}",
                f"{configs['val_acc']:.2f}"
            ])

        return Just(configs)
    except Exception as e:
        return Maybe(f"Logging failed: {e}", monoid=False)


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def create_model(configs: dict) -> Maybe:
    """Create and initialize the CNN model."""
    try:
        model_settings = configs.get("model_settings", {})
        num_classes = model_settings.get("num_classes", 10)
        dropout_rate = model_settings.get("dropout_rate", 0.5)

        device = get_device(model_settings.get("device", "auto"))

        model = CNN(num_classes=num_classes, dropout_rate=dropout_rate)
        model = model.to(device)

        # Add model and device to configs
        configs["model"] = model
        configs["device"] = device

        print(f"Model created and moved to {device}")
        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Failed to create model: {str(e)}",monoid=False)

def create_optimizer(configs: dict) -> Maybe:
    """Create optimizer for training."""
    try:
        model = configs.get("model")
        if model is None:
            return Maybe("Model not found in configs", monoid=False)

        model_settings = configs.get("model_settings", {})
        learning_rate = model_settings.get("learning_rate", 0.001)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        configs["optimizer"] = optimizer

        print(f"Optimizer created with lr={learning_rate}")
        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Failed to create optimizer: {str(e)}",monoid=False)

def create_loss_function(configs: dict) -> Maybe:
    """Create loss function for training."""
    try:
        # For classification, CrossEntropyLoss is standard
        criterion = nn.CrossEntropyLoss()
        configs["criterion"] = criterion

        print("Loss function: CrossEntropyLoss")
        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Failed to create loss function: {str(e)}",monoid=False)

    ###########################################################################
    #                           Training/evaluation                           #
    ###########################################################################

def train_one_epoch(configs: dict) -> Maybe:
    """Train for one epoch."""
    try:
        model = configs["model"]
        train_loader = configs["train_loader"]
        optimizer = configs["optimizer"]
        criterion = configs["criterion"]
        device = configs["device"]
        epoch = configs.get("current_epoch", 0)

        model.train()  # Set to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader),
                                                total=len(train_loader),
                                                desc=f"Epoch {epoch}"):
            #for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 10 batches
            # if batch_idx % 10 == 0:
            #     print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(train_loader)}], '
            #           f'Loss: {loss.item():.4f}')

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f'\nEpoch [{epoch}] Summary:')
        print(f'  Average Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%\n')

        configs["epoch_loss"] = epoch_loss
        configs["epoch_acc"] = epoch_acc

        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Training failed: {str(e)}", monoid=False)



def validate_one_epoch(configs: dict) -> Maybe:
    """Validate for one epoch."""
    try:
        model = configs["model"]
        val_loader = configs["val_loader"]
        criterion = configs["criterion"]
        device = configs["device"]
        epoch = configs.get("current_epoch", 0)

        model.eval()  # Set to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradients needed for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f'Validation Results:')
        print(f'  Loss: {val_loss:.4f}')
        print(f'  Accuracy: {val_acc:.2f}%\n')

        configs["val_loss"] = val_loss
        configs["val_acc"] = val_acc
        if configs["val_acc"] > configs["best_val_acc"]:
            configs["best_val_acc"] = configs["val_acc"]
            save_model(configs)

        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Validation failed: {str(e)}",monoid=False)



def training_loop(configs: dict) -> Maybe:
    """Main training loop."""
    try:
        model_settings = configs.get("model_settings", {})
        num_epochs = model_settings.get("num_epochs", 10)
        configs["best_val_acc"] = -1.0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            configs["current_epoch"] = epoch

            # Train one epoch
            result = train_one_epoch(configs)
            if result.is_nothing():
                return result

            # Validate one epoch
            result = validate_one_epoch(configs)
            if result.is_nothing():
                return result
            result = log_metrics(configs)
        print("Training completed!")
        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Training loop failed: {str(e)}",monoid=False)

    ###########################################################################
    #                                Save/load                                #
    ###########################################################################
def save_model(configs: dict) -> Maybe:
    """Save model checkpoint."""
    try:
        model = configs["model"]
        save_path = Path(configs["model_settings"].get("save_path", "model.pt"))

        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': configs.get("val_acc", 0),
            'epoch': configs.get("current_epoch", 0)
        }, save_path)

        print(f"Model saved to {save_path}")
        return Just(configs)
    except Exception as e:
        return Maybe(f"Save failed: {e}", monoid=False)


def load_model(configs: dict) -> Maybe:
    """Load a pre-trained model state from a given path."""
    try:
        model = configs.get("model")
        device = configs.get("device") # Ensure device is available

        if model is None:
            return Maybe(value="Model not found in configs before loading state.", monoid=False)
        if device is None:
            return Maybe(value="Device not found in configs for model loading.", monoid=False)

        model_path = configs.get("model_path") # This should be set by the calling function (e.g., run_inference)
        if model_path is None:
            return Maybe(value="Model path not provided in configs.", monoid=False)

        model_file = Path(model_path)
        if not model_file.exists():
            return Maybe(value=f"Model file not found at {model_path}.", monoid=False)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to evaluation mode for inference

        print(f"Model loaded from {model_path} and set to evaluation mode.")
        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Failed to load model: {str(e)}", monoid=False)

    ###########################################################################
    #                                Inference                                #
    ###########################################################################
def inference_loop(configs: dict) -> Maybe:
    """Main inference loop for evaluating a loaded model."""
    print("Running inference loop...")
    return config_helper.inference_evaluation(configs)
