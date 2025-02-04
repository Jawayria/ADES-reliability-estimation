import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torchmetrics.classification import (BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy, BinaryAUROC)
import os
import csv
from pathlib import Path



def plot_rel_distribution(all_rels: list[int]):
    """
    Plots the distribution of classes in the dataset.
    Args:
    - all_rels: A list of integers representing classes.
    """
    # Count occurrences of each class
    class_counts = Counter(all_rels)
    classes = sorted(class_counts.keys())
    frequencies = [class_counts[cls] for cls in classes]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(classes, frequencies, edgecolor='k', alpha=0.7, color='skyblue')
    plt.title("Distribution of Classes in the Dataset", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(classes)  # Ensure all classes are shown on the x-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_matrix(true_values, predicted_values, accuracy, model_name):
  cm = confusion_matrix(true_values, predicted_values)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # Replace with your class labels if needed
  disp.plot(cmap=plt.cm.Blues)

  plt.title(f'Confusion Matrix for {model_name} - Accuracy: {accuracy:.2f}')
  plt.show()

def generate_metrics(true_values, predicted_values, match, model_checkpoints_path, num_epochs, learning_rate, node_features, dropout_rate, patience, hidden_dim):
    torch_true_values = torch.tensor(true_values)
    torch_predicted_values = torch.tensor(predicted_values)

    # Initialize metrics
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()
    auroc = BinaryAUROC()
    accuracy = BinaryAccuracy()

    # Compute metrics
    precision_val = precision(torch_predicted_values, torch_true_values)
    recall_val = recall(torch_predicted_values, torch_true_values)
    f1_val = f1(torch_predicted_values, torch_true_values)
    auroc_val = auroc(torch_predicted_values.float(), torch_true_values)
    accuracy_val = accuracy(torch_predicted_values, torch_true_values)

    # Print results
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall: {recall_val:.4f}")
    print(f"F1 Score: {f1_val:.4f}")
    print(f"AUROC: {auroc_val:.4f}")
    print(f"Accuracy: {accuracy_val:.4f}")


    data = {
        "Match": match,
        "Number of epochs": num_epochs,
        "Learning rate": learning_rate,
        "Node features": node_features,
        "Dropout rate": dropout_rate,
        "Patience": patience,
        "Hidden dimension": hidden_dim,
        "Precision": precision_val.item(),
        "Recall": recall_val.item(),
        "F1 Score": f1_val.item(),
        "Accuracy": accuracy_val.item(),
        "AUROC": auroc_val.item()
    }

    file_path = f"{model_checkpoints_path}/booster/results/{match}.csv"

    write_header = not Path(file_path).exists()  # Add header only if the file does not exist

    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if write_header:
            writer.writeheader()  # Write header if the file is being created for the first time

        writer.writerow(data)

    print(f"Metrics saved to {file_path}")