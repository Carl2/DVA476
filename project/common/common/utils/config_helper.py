#!/usr/bin/env python
from pymonad.maybe import Maybe,Just
import torch
import torch.optim as optim
import torch.nn as nn
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from torchmetrics import ConfusionMatrix, Accuracy
import csv


class DatasetType(Enum):
    LOGGING = 0
    TRAIN = 1
    VALIDATE = 2
    TEST = 3
    MODEL_SETTINGS = 4
    SETTINGS = 5





def get_dataset_config(config: dict, dataset_type: DatasetType):
    """Get the configuration dictionary for a specific dataset type."""
    if dataset_type == DatasetType.SETTINGS:
        return config.get("settings", None)
    elif dataset_type == DatasetType.TRAIN:
        return config.get("train_settings", None)
    elif dataset_type == DatasetType.VALIDATE:
        return config.get("validation_settings", None)
    elif dataset_type == DatasetType.TEST:
        return config.get("test_settings", None)
    elif dataset_type == DatasetType.MODEL_SETTINGS:
        return config.get("model_settings", None)
    elif dataset_type == DatasetType.LOGGING:
        return config.get("logging", None)
    return None

def get_key_fn(config: dict, dataset_type: DatasetType) -> callable:
    """Create a function to retrieve keys from the appropriate config section."""
    type_config = get_dataset_config(config, dataset_type)

    def do_get_key_fn(key: str) -> Maybe:
        if type_config is None:
            return Maybe(
                value=f"Config section not found for {dataset_type.name}",
                monoid=False
            )

        val = type_config.get(key, None)
        if val is None:
            return Maybe(
                value=f"Key '{key}' not found in {dataset_type.name}",
                monoid=False
            )

        return Just(val)

    return do_get_key_fn


def convert_to_key(dataset_type: DatasetType):
    if dataset_type == DatasetType.TRAIN:
        return Just("train_settings")
    elif dataset_type == DatasetType.VALIDATE:
        return Just("validation_settings")
    elif dataset_type == DatasetType.TEST:
        return Just("test_settings")
    return Maybe(value=f"dataset type is not convertible ", monoid=False)




def _get_from_config(*,dataset: DatasetType, config):
    if dataset == DatasetType.SETTINGS:
        return config['settings']
    if dataset == DatasetType.TRAIN:
        return config['train']
    if dataset == DatasetType.VALIDATE:
        return config['validate']
    if dataset == DatasetType.TEST:
        return config['test']





def get_settings(config: dict, config_type: DatasetType) -> callable:
    config = _get_from_config(dataset=config_type, config=config)

    def do_get_train_setting(key_str) -> Maybe:
        try:
            return Just(config[key_str])
        except Exception as e:
            return Maybe(value=f"Unable to find setting {e}", monoid=False)
    return do_get_train_setting


def get_classes(config: dict, config_type: DatasetType)->list:
    _config = _get_from_config(dataset=config_type, config=config)
    return list(_config.keys())


def get_start_end(config_specific: dict):
    return (config_specific['start'],config_specific['end'])

###############################################################################
#                                 Log metrics                                 #
###############################################################################
def log_metrics(logging_fn:callable) -> callable:
    m_csv_file  = logging_fn("metrics_csv")
    m_every_n  = logging_fn("plot_every_n_epochs")
    m_path = logging_fn("save_plots_to")

    if (m_csv_file.is_nothing() or
        m_every_n.is_nothing() or
        m_path.is_nothing()):
        print(f"Logging was not able to find configuration")
        return lambda config: f"No metrics saved"

    csv_path = Path(m_csv_file.value).expanduser().resolve()
    every_n = m_every_n.value
    plot_path = Path(m_path.value).expanduser().resolve()
    print(f"🐜 metrics are saved on {csv_path}")


    def do_log_metrics(configs: dict) -> Maybe:
        try:

            # Create directory if needed
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists to write header
            file_exists = csv_path.exists()

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
            print(f"💀 Failed logging {e}")
            return Maybe(f"Logging failed: {e}", monoid=False)
    return do_log_metrics



###############################################################################
#                                Optimizer/loss                               #
###############################################################################

def create_optimizer(model_settings_fn:callable) -> callable:
    """Create optimizer for training."""


    def do_create_optimizer(config) -> Maybe:
        model = config.get("model",None)
        if model is None:
            return Maybe("Model not found in configs", monoid=False)

        maybe_learning_rate = model_settings_fn("learning_rate")

        if maybe_learning_rate.is_nothing():
            return Maybe(value=f"Could not find learning in model settings", monoid=False)
        learning_rate = maybe_learning_rate.value

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        config["optimizer"] = optimizer

        print(f"Optimizer created with lr={learning_rate}")

        return Just(config)
    return do_create_optimizer


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
#                           Training/evaluation for cnn_model
#  This is actually pretty much the same as the CNN network, should have
# shared , but time is of an essence  :(
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



def validate_one_epoch(configs: dict, save_fn: callable) -> Maybe:
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
            save_fn(configs)

        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Validation failed: {str(e)}",monoid=False)




def training_loop(model_settings_fn: callable,
                  settings_fn: callable,
                  logging_fn: callable) -> callable:
    print(f"Start training")
    save_fn = save_model(settings_fn, model_settings_fn)
    logging_fn = log_metrics(logging_fn)


    def do_training_loop(configs: dict) -> Maybe:
        """Main training loop."""
        try:

            configs["device"] = model_settings_fn("device").value #Short hack
            maybe_num_epochs = model_settings_fn("num_epochs")
            if maybe_num_epochs.is_nothing():
                return Maybe(value=f"Number of epochs is not defined", monoid=False)

            num_epochs = maybe_num_epochs.value

            configs["best_val_acc"] = -1.0

            print(f"Starting training for {num_epochs} epochs...")

            for epoch in range(1, num_epochs + 1):
                configs["current_epoch"] = epoch

                # Train one epoch
                result = train_one_epoch(configs)
                if result.is_nothing():
                    return result

                # Validate one epoch
                result = validate_one_epoch(configs, save_fn)
                if result.is_nothing():
                    return result
                result = logging_fn(configs)
            print("Training completed!")
            return Just(configs)
        except Exception as e:
            return Maybe(value=f"Training loop failed: {str(e)}",monoid=False)

    return do_training_loop


###############################################################################
#                                  Save model                                 #
###############################################################################
def save_model(setting_fn: callable, model_settings_fn: callable ) -> callable:

    m_mean = setting_fn("mean")
    m_std = setting_fn("std")
    m_save_path = model_settings_fn('save_path')

    if m_mean.is_nothing() or m_std.is_nothing() or m_save_path.is_nothing():
        print(f"Unable to get mean or std for saving!")
        return lambda config: print(f"Could not save {m_save_path} ")
    mean = m_mean.value
    std = m_std.value
    save_path = Path(m_save_path.value).expanduser().resolve()


    def do_save_model(configs):
        try:
            model = configs["model"]
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': configs.get("val_acc", 0),
                'epoch': configs.get("current_epoch", 0),
                'mean': mean,
                'std': std
            }, save_path)

            print(f"Model saved to {save_path}")
            return Just(configs)
        except Exception as e:
            print(f"Could not save {e} {save_path}")

            return Maybe(f"Save failed: {e}", monoid=False)
    return do_save_model


def load_model(config: dict) -> Maybe:
    """Load a model from a saved state and update config."""
    try:
        load_path = config.get('load_model',None)
        if load_path is None:
            return Maybe(value="Load path not found in model settings", monoid=False)

        if not load_path.exists():
            return Maybe(value=f"Model file not found at {load_path}", monoid=False)

        checkpoint = torch.load(load_path)

        # Update model state dict
        model = config.get("model")
        if model is None:
             return Maybe(value="Model instance not found in config. Please create a model before loading.", monoid=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        config['model'] = model

        # Update settings for mean and std if they exist in checkpoint
        if 'mean' in checkpoint and 'std' in checkpoint:
            if 'settings' not in config:
                config['settings'] = {}
            config['settings']['mean'] = checkpoint['mean']
            config['settings']['std'] = checkpoint['std']
            print(f"Loaded mean: {checkpoint['mean']}, std: {checkpoint['std']}")

        print(f"Model loaded successfully from {load_path}")
        return Just(config)
    except Exception as e:
        return Maybe(value=f"Failed to load model: {str(e)}", monoid=False)



def inference_evaluation(configs: dict) -> Maybe:
    """Evaluate the model on a given loader (e.g., test set)."""
    try:
        model = configs["model"]
        data_loader = configs["dataloader"]
        criterion = configs["criterion"]
        device = configs["device"]
        num_classes = configs.get("num_classes", 10)

        # This will be used for confusion matrix.
        all_preds = []
        all_labels = []

        model.eval()  # Set to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0
        metric_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)


        print("Starting inference evaluation...")
        with torch.no_grad():  # No gradients needed for inference
            for images, labels in tqdm(data_loader, desc="Evaluating model"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                metric_accuracy.update(predicted, labels)
                # store the predicted and the labels.
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

        inference_loss = running_loss / len(data_loader)
        inference_acc = 100 * correct / total


        # Compute the confusion matrix
        # Concatenate all collected predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # compute the final accuracy.
        inference_acc_torchmetrics = metric_accuracy.compute().item() * 100

        # Create ConfusionMatrix object and compute
        confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        confusion_matrix = confmat_metric(all_preds, all_labels)
        configs["confusion_matrix"] = confusion_matrix.numpy()


        print(f'\n--- Inference Results ---')
        print(f'  Total samples: {total}')
        print(f'  Loss: {inference_loss:.4f}')
        print(f'  Accuracy: {inference_acc:.2f}%\n')
        print(f'  Accuracy (torchmetrics): {inference_acc_torchmetrics:.2f}%\n')
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix)


        configs["inference_loss"] = inference_loss
        configs["inference_acc"] = inference_acc

        return Just(configs)
    except Exception as e:
        return Maybe(value=f"Inference evaluation failed: {str(e)}", monoid=False)
