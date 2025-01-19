import numpy as np
import torch

def evaluate(device, model, test_loader, best_model_name ):
  print(model)
  # Load the saved state dictionary
  model.load_state_dict(torch.load(best_model_name, weights_only=True))

  # Evaluation code
  true_values = []
  predicted_values = []

  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_loader:
          data = data.to(device)
          out = model(data)
          pred = out.argmax(dim=1)  # Get predicted class
          correct += (pred == data.y).sum().item()
          total += data.y.size(0)

          true_values.extend(data.y.cpu().numpy())  # Convert to NumPy array
          predicted_values.extend(pred.cpu().numpy())

  print(true_values[:10])
  print(predicted_values[:10])


  true_values = np.array(true_values)
  predicted_values = np.array(predicted_values)
  accuracy = correct / total

  print(f'Accuracy: {accuracy:.4f}')

  return true_values, predicted_values, accuracy