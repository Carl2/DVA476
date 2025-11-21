#!/usr/bin/env python
import matplotlib.pyplot as plt


def TrainPlot(history: dict):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    # Loss plot
    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history["epoch"], history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Confidence plot
    ax3.plot(history["epoch"], history["avg_confidence"], label="Avg Confidence", color="orange")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Confidence")
    ax3.set_title("Average Prediction Confidence")
    ax3.set_ylim([0, 1])  # Confidence is between 0 and 1
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns




def test_plot(history: dict, class_names=None):
    """
    Create comprehensive visualizations for test results.

    Args:
        history: Dictionary containing:
            - 'labels': Ground truth labels (tensor or array)
            - 'predicted': Predicted labels (tensor or array)
            - 'accuracy': Overall accuracy (float)
            - 'confidence': Average confidence (float)
        class_names: Optional list of class names for labeling
    """
    # Convert tensors to numpy if needed
    if hasattr(history['labels'], 'cpu'):
        labels = history['labels'].cpu().numpy()
        predicted = history['predicted'].cpu().numpy()
    else:
        labels = np.array(history['labels'])
        predicted = np.array(history['predicted'])

    # Get number of classes
    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f'{i}' for i in range(num_classes)]

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 2, 1)
    cm = confusion_matrix(labels, predicted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title(f'Confusion Matrix\nOverall Accuracy: {history["accuracy"]:.4f}')

    # 2. Per-Class Accuracy
    ax2 = plt.subplot(2, 2, 2)
    per_class_acc = []
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (predicted[mask] == labels[mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0)

    bars = ax2.bar(range(num_classes), per_class_acc, color='steelblue', alpha=0.7)
    ax2.axhline(y=history['accuracy'], color='red', linestyle='--',
                label=f'Overall Acc: {history["accuracy"]:.4f}')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Prediction Distribution
    ax3 = plt.subplot(2, 2, 3)
    unique_pred, counts_pred = np.unique(predicted, return_counts=True)
    unique_true, counts_true = np.unique(labels, return_counts=True)

    x = np.arange(num_classes)
    width = 0.35

    # Ensure all classes are represented
    pred_counts = np.zeros(num_classes)
    true_counts = np.zeros(num_classes)
    pred_counts[unique_pred] = counts_pred
    true_counts[unique_true] = counts_true

    ax3.bar(x - width/2, true_counts, width, label='True', alpha=0.7, color='green')
    ax3.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='orange')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of True vs Predicted Labels')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Metrics Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Calculate precision, recall, F1 per class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, average=None, zero_division=0
    )

    # Create text summary
    summary_text = f"Test Results Summary\n{'='*40}\n\n"
    summary_text += f"Overall Accuracy: {history['accuracy']:.4f}\n"
    summary_text += f"Avg Confidence: {history['confidence']:.4f}\n"
    summary_text += f"Total Samples: {len(labels)}\n\n"
    summary_text += f"Per-Class Metrics:\n{'-'*40}\n"

    for i in range(num_classes):
        summary_text += f"\n{class_names[i]}:\n"
        summary_text += f"  Precision: {precision[i]:.4f}\n"
        summary_text += f"  Recall:    {recall[i]:.4f}\n"
        summary_text += f"  F1-Score:  {f1[i]:.4f}\n"
        summary_text += f"  Support:   {support[i]}\n"

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()

    # Print classification report to console
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(labels, predicted, target_names=class_names))

    return fig
