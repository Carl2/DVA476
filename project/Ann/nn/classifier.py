#!/usr/bin/env python
from functools import reduce
from pdb import set_trace
from pathlib import Path

import torch
import torch.nn as nn
from pymonad.maybe import Just, Maybe, Nothing

from nn.early_stopper import EarlyStopping

class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(8261, 512)      # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)       # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 100)       # Output layer

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)                      # No activation; use CrossEntropyLoss
        return x


def train_model(model: nn.Module, **kwargs) -> callable:
    def do_train_model(batch) -> Maybe:
        labels = batch['labels']
        feature_data = batch['normalized']
        criterion = kwargs.get('loss')
        optimizer = kwargs.get('optimizer')
        epochs = kwargs.get("epochs")
        label_tensor = torch.tensor(labels, dtype=torch.long)
        validation_label = kwargs.get("validation_label")
        validation_data = kwargs.get("validation_data")

        # history dict
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "avg_confidence": []}

        # Early stopper to prevent overfitting.
        early_stopping = EarlyStopping(
            patience=kwargs.get('patience', 10),
            min_delta=kwargs.get('min_delta', 0.001),
            verbose=False
        )

        assert len(feature_data) == len(labels), f"Length mismatch: training={len(feature_data)}, Y_train={len(labels)}"
        assert len(validation_data) == len(validation_label), f"Length mismatch: validation={len(validation_data)}, Y_train={len(validation_label)}"

        feature_tensor = torch.nan_to_num(feature_data, nan=0.0)
        validation_tensor = torch.nan_to_num(validation_data, nan=0.0)

        for epoch in range(epochs):
            outputs = model(feature_tensor)
            loss = criterion(outputs, label_tensor)

            # Backward pass
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Calculate new gradients
            optimizer.step()       # Update weights

            model.eval()           # Set in evaluation mode
            with torch.no_grad():  # Disable gradient calculation
                val_outputs = model(validation_tensor)
                val_loss = criterion(val_outputs, torch.tensor(validation_label,
                                                               dtype=torch.long))

                # Get the index value , with the highest probability.
                # Max returns (values,indices) for all validations
                # a val_outputs = [[....],[...],[...]..,..] of each validation
                # Applying softmax will rescale it to [0,1] for each output
                probabilities = torch.softmax(val_outputs, dim=1)
                # doing max on dimension 1, will get a the highest value,index for each row [0,2,3,...]
                # one for each val_output.
                # - max_values: the highest probability for each sample [0.9, 0.85, ...]
                # - predicted: the class index with that probability [2, 0, 3, ...]
                max_values, predicted = torch.max(probabilities, dim=1)
                # The accuracy, we now take the validationlabel and compare
                # the predicted with the validations_labels and convert it to float.
                # so  [1,3,5,1] == [1,2,5,0] = [True,False,True,False] -> [1.0,0.0,1.0,0.0]
                # and the mean of that would be 50%
                accuracy = (predicted == torch.tensor(validation_label)).float().mean()

                avg_confidence = max_values.mean().item()

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(accuracy.item())
            history["avg_confidence"].append(avg_confidence)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {loss.item():.4f}, '
                      f'Val Loss: {val_loss.item():.4f}, '
                      f'Val Acc: {accuracy.item()*100:.2f}%, '
                      f'avg confidence: {avg_confidence*100:.2f}%'
                      )
            if early_stopping(val_loss.item(), model):
                print(f"\n🛑 Stopping early at epoch {epoch+1}")
                break

        early_stopping.load_best_model(model)
        return Just((history, early_stopping.best_model_state, batch['stats']))
    return do_train_model


def save_model(model_name_path: Path) -> callable:
    def do_save_model(history_model) -> Maybe:
        history, best_state, norm_stats = history_model
        torch.save({
            'model_state_dict': best_state,
            'norm_mean': norm_stats['mean'],
            'norm_std': norm_stats['std']
        }, model_name_path)
        print(f"✓ Model + normalization stats saved to {model_name_path}")
        return Just(history_model[0])
    return do_save_model


# TODO:
def load_model(model_data_path: Path) -> callable:
    """Return a function that loads model and runs inference"""

    def do_load_model(batch: dict) -> Maybe:
        labels = batch['labels']
        test_data_raw = batch['tensor']
        # Load checkpoint
        checkpoint = torch.load(model_data_path, map_location='cpu')

        # Create model
        model = ClassifierNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load weights with error checking
        norm_stats = {
            'mean': checkpoint['norm_mean'],
            'std': checkpoint['norm_std']
        }

        # Normalize test data with training stats
        test_data = (test_data_raw - norm_stats['mean']) / norm_stats['std']
        test_data = torch.nan_to_num(test_data, nan=0.0)

        try:
            missing, unexpected = model.load_state_dict(
                checkpoint['model_state_dict'],  # ← Use the NESTED dict
                strict=True
            )

            if missing or unexpected:
                return Maybe(
                    value=f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}",
                    monoid=False
                )
        except Exception as e:
            return Maybe(value=f"Failed to load model: {e}", monoid=False)

        print(f"\n🔍 Model Debug Info:")
        print(f"  First layer weights mean: {model.fc1.weight.mean().item():.6f}")
        print(f"  First layer weights std: {model.fc1.weight.std().item():.6f}")

        model.eval()
        print(f"✓ Model loaded from {model_data_path}")

        # Ensure data is on same device
        test_data = test_data.cpu()
        labels = torch.tensor(labels, dtype=torch.long).cpu()

        assert len(test_data) == len(labels), \
            f"Length mismatch: test={len(test_data)}, labels={len(labels)}"

        with torch.no_grad():
            test_output = model(test_data)
            probabilities = torch.softmax(test_output, dim=1)
            max_values, predicted = torch.max(probabilities, dim=1)
            accuracy = (predicted == labels).float().mean()
            avg_confidence = max_values.mean().item()

        return Just({
            "labels": labels.tolist(),
            "predicted": predicted.tolist(),
            "accuracy": accuracy.item(),
            "confidence": avg_confidence
        })
    return do_load_model



# def load_model(model_data_path: Path, model_class, **kwargs) -> callable:

#     model = model_class() if callable(model_class) else model_class

#     model.load_state_dict(torch.load(model_data_path, map_location='cpu'))
#     model.eval() # eval mode
#     print(f"✓ Model loaded from {model_data_path}")



#     def do_load_model(batch) -> Maybe:
#         history = {}
#         labels, test_data = batch
#         assert len(test_data) == len(labels), f"Length mismatch: test data={len(test_data)}, labels={len(labels)}"
#         test_output = model(test_data)
#         # Do more later
#         probabilities = torch.softmax(test_output, dim=1)
#         max_values, predicted = torch.max(probabilities, dim=1)
#         accuracy = (predicted == torch.tensor(labels)).float().mean()
#         avg_confidence = max_values.mean().item()
#         return Just({
#             "labels": labels,
#             "predicted": predicted,
#             "accuracy": accuracy,
#             "confidence": avg_confidence
#         })

#     return do_load_model
