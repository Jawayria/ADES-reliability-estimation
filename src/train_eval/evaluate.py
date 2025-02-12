import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models.GAT import GAT


def evaluate(device, model, test_loader, best_model_name):
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


def evaluate_multiclass(device, model, test_loader, best_model_name):
    true_values, predicted_values, accuracy = evaluate(device, model, test_loader, best_model_name)
    squared_distance = np.sum((true_values - predicted_values) ** 2)
    rmse = np.sqrt(squared_distance / len(true_values))
    percentage_outside_rad_1 = np.sum(np.abs(true_values - predicted_values) > 1) / len(true_values)
    percentage_outside_rad_2 = np.sum(np.abs(true_values - predicted_values) > 2) / len(true_values)
    return true_values, predicted_values, squared_distance, accuracy, rmse, percentage_outside_rad_1, percentage_outside_rad_2


def evaluate_ensamble(device, ensemble: dict[str, GAT], test_loader: DataLoader) -> tuple[
    np.ndarray, np.ndarray, int, float, float, float, float]:
    predictions = []
    true_classes = []

    for match, model in ensemble.items():
        model.eval()

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            match = "1-1"
            out = ensemble[match](data)
            pred = out.argmax(dim=1)  # Get predicted class
            true_classes.extend(data.y.cpu().numpy())

            if pred == 0:
                match = "2-1"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    match = "3-1"
                    out = ensemble[match](data)
                    pred = out.argmax(dim=1)
                    if pred == 0:
                        predictions.append(0)
                    else:
                        predictions.append(1)
                else:
                    match = "3-2"
                    out = ensemble[match](data)
                    pred = out.argmax(dim=1)
                    if pred == 0:
                        predictions.append(2)
                    else:
                        predictions.append(3)
            else:
                match = "2-2"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    match = "3-3"
                    out = ensemble[match](data)
                    pred = out.argmax(dim=1)
                    if pred == 0:
                        predictions.append(4)
                    else:
                        predictions.append(5)
                else:
                    match = "3-4"
                    out = ensemble[match](data)
                    pred = out.argmax(dim=1)
                    if pred == 0:
                        predictions.append(6)
                    else:
                        predictions.append(7)
    sq_distance = np.sum((np.array(true_classes) - np.array(predictions)) ** 2)
    rmse = np.sqrt(sq_distance / len(true_classes))
    accuracy = np.sum(np.array(true_classes) == np.array(predictions)) / len(true_classes)
    percentage_outside_rad_1 = np.sum(np.abs(np.array(true_classes) - np.array(predictions)) > 1) / len(true_classes)
    percentage_outside_rad_2 = np.sum(np.abs(np.array(true_classes) - np.array(predictions)) > 2) / len(true_classes)
    return np.array(true_classes), np.array(
        predictions), sq_distance, accuracy, rmse, percentage_outside_rad_1, percentage_outside_rad_2
