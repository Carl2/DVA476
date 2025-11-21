#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pymonad.maybe import Maybe, Just
from pathlib import Path
import pandas as pd

def _get_class_names(config: dict):
    """
    Helper function to extract class names from config.
    Returns list of class names or numeric labels.
    """
    dataloader = config.get('dataloader')
    if dataloader is None:
        raise ValueError("Dataloader not found in config")

    dataset = dataloader.dataset

    # Try to get class names, fall back to numeric labels
    if hasattr(dataset, 'class_to_idx') and dataset.class_to_idx:
        sorted_items = sorted(dataset.class_to_idx.items(), key=lambda item: item[1])
        return [name for name, index in sorted_items]
    else:
        num_classes = config['confusion_matrix'].shape[0]
        return [str(i) for i in range(num_classes)]


def plot_confusion_matrix(cm, class_names=None, figsize=(8,6),
                          cmap="Blues", fmt="d", title="Confusion matrix",
                          save_path=None):
    """
    cm: numpy array shape (C,C) (rows=actual, cols=predicted)
    class_names: list of length C or None
    fmt: "d" for integer counts, ".2f" for normalized values
    """
    cm = np.array(cm)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_normalized_confusion_matrix(config: dict) -> Maybe:
    """
    Plot confusion matrix normalized by true labels (rows sum to 1).
    """
    try:
        cm = config['confusion_matrix']
        class_names = _get_class_names(config)

        # Normalize confusion matrix (rows sum to 1)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        print("Plotting normalized confusion matrix...")
        plot_confusion_matrix(
            cm=cm_normalized,
            class_names=class_names,
            fmt=".2f",
            title="Normalized Confusion Matrix (by True Label)",
            save_path="confusion_matrix_normalized.png"
        )

        return Just(config)

    except Exception as e:
        return Maybe(f"Failed to plot normalized confusion matrix: {e}", monoid=False)


def plot_per_class_metrics(config: dict) -> Maybe:
    """
    Plot per-class precision, recall, and F1-score as bar charts.
    """
    try:
        cm = config['confusion_matrix']
        class_names = _get_class_names(config)

        # Calculate per-class metrics
        precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
        recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        print("Plotting per-class metrics...")

        # Create bar plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        x = np.arange(len(class_names))

        axes[0].bar(x, precision)
        axes[0].set_title('Precision per Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x, recall)
        axes[1].set_title('Recall per Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        axes[2].bar(x, f1)
        axes[2].set_title('F1-Score per Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(class_names, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig("per_class_metrics.png", dpi=200)
        plt.show()

        return Just(config)

    except Exception as e:
        return Maybe(f"Failed to plot per-class metrics: {e}", monoid=False)


def plot(config: dict) -> Maybe:
    """
    Plots the regular confusion matrix.
    """
    try:
        class_names = _get_class_names(config)

        print("Plotting confusion matrix...")
        plot_confusion_matrix(
            cm=config['confusion_matrix'],
            class_names=class_names,
            title="Inference Confusion Matrix",
            save_path="confusion_matrix.png"
        )
        plot_per_class_metrics(config)
        return Just(config)

    except Exception as e:
        return Maybe(f"Failed to plot confusion matrix: {e}", monoid=False)


def plot_training_history(csv_path: str) -> callable:
    def do_plot_training_history(config: dict) -> Maybe:
        """
        Reads the training log CSV and plots accuracy and loss curves.
        """
        try:
            if csv_path is None:
                return Maybe("metrics_csv path not found in config", monoid=False)

            if not Path(csv_path).exists():
                print(f"Warning: Log file not found at {csv_path}. Skipping history plot.")
                return Just(config)

            # Read the data using pandas
            history_df = pd.read_csv(csv_path)

            if history_df.empty:
                print("Warning: Log file is empty. Skipping history plot.")
                return Just(config)

            print("Plotting training and validation history...")

            # Create a figure with two subplots (side-by-side)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # --- Plot 1: Accuracy ---
            ax1.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy', marker='o')
            ax1.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy', marker='o')
            ax1.set_title('Accuracy vs. Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True)

            # --- Plot 2: Loss ---
            ax2.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
            ax2.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', marker='o')
            ax2.set_title('Loss vs. Epochs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig("training_history.png", dpi=300)
            plt.show()

            return Just(config)

        except Exception as e:
            return Maybe(f"Failed to plot training history: {e}", monoid=False)
    return do_plot_training_history
