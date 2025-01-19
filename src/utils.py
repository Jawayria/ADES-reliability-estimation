import matplotlib.pyplot as plt
import re

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_config_id(filename):
    match = re.search(r"config_(\d+)", filename)
    return int(match.group(1)) if match else None

def generate_matrix(true_values, predicted_values, accuracy, model_name):
  cm = confusion_matrix(true_values, predicted_values)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # Replace with your class labels if needed
  disp.plot(cmap=plt.cm.Blues)

  plt.title(f'Confusion Matrix for {model_name} - Accuracy: {accuracy:.2f}')
  plt.show()