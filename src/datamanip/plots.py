import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
