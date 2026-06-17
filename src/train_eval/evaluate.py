import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch_geometric.loader import DataLoader
from torchmetrics.functional import f1_score

from models.GAT import GAT


def evaluate(device, model, test_loader, best_model_name):
    print(model)
    # Load the saved state dictionary
    #model.load_state_dict(torch.load(best_model_name, weights_only=True))

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
    true_values, predicted_values, accuracy = evaluate(
        device,
        model,
        test_loader,
        best_model_name
    )

    squared_distance = np.sum((true_values - predicted_values) ** 2)
    rmse = np.sqrt(squared_distance / len(true_values))

    ordinal_mae = np.mean(np.abs(true_values - predicted_values))

    qwk = cohen_kappa_score(
        true_values,
        predicted_values,
        weights="quadratic"
    )

    percentage_outside_rad_1 = (
        np.sum(np.abs(true_values - predicted_values) > 1)
        / len(true_values)
    )

    percentage_outside_rad_2 = (
        np.sum(np.abs(true_values - predicted_values) > 2)
        / len(true_values)
    )

    return (
        true_values,
        predicted_values,
        squared_distance,
        accuracy,
        rmse,
        ordinal_mae,
        qwk,
        percentage_outside_rad_1,
        percentage_outside_rad_2,
    )

def evaluate_ensemble(device, ensemble: dict[str, GAT], test_loader: DataLoader) -> tuple[
    np.ndarray, np.ndarray, int, float, float, float, float, float]:
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
    f1 = f1_score(torch.tensor(true_classes), torch.tensor(predictions), num_classes=8, task="multiclass")
    return np.array(true_classes), np.array(
        predictions), sq_distance, accuracy, rmse, percentage_outside_rad_1, percentage_outside_rad_2, f1

def evaluate_soft_voting_ensemble(device, ensemble: dict[str, GAT], test_loader: DataLoader) -> tuple[
    np.ndarray, np.ndarray, int, float, float, float, float, float]:
    predictions = []
    true_classes = []

    # Set all models to evaluation mode.
    for model in ensemble.values():
        model.eval()

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            true_classes.extend(data.y.cpu().numpy())

            # Get softmax probabilities from all 7 binary classifiers.
            prob_1_1 = torch.softmax(ensemble["1-1"](data), dim=1)  # [batch_size, 2]
            prob_2_1 = torch.softmax(ensemble["2-1"](data), dim=1)
            prob_3_1 = torch.softmax(ensemble["3-1"](data), dim=1)
            prob_3_2 = torch.softmax(ensemble["3-2"](data), dim=1)
            prob_2_2 = torch.softmax(ensemble["2-2"](data), dim=1)
            prob_3_3 = torch.softmax(ensemble["3-3"](data), dim=1)
            prob_3_4 = torch.softmax(ensemble["3-4"](data), dim=1)

            # Compute aggregated probability for each final class (0 to 7)
            # Path breakdown:
            # Classes 0 & 1: branch "1-1" (output 0) then "2-1" (output 0) then "3-1"
            p_branch1 = prob_1_1[:, 0]
            p_2_1_0 = prob_2_1[:, 0]
            # Classes 0 and 1 split in "3-1"
            prob_class_0 = p_branch1 * p_2_1_0 * prob_3_1[:, 0]
            prob_class_1 = p_branch1 * p_2_1_0 * prob_3_1[:, 1]

            # Classes 2 & 3: branch "1-1" (output 0) then "2-1" (output 1) then "3-2"
            p_2_1_1 = prob_2_1[:, 1]
            prob_class_2 = p_branch1 * p_2_1_1 * prob_3_2[:, 0]
            prob_class_3 = p_branch1 * p_2_1_1 * prob_3_2[:, 1]

            # Classes 4 & 5: branch "1-1" (output 1) then "2-2" (output 0) then "3-3"
            p_branch2 = prob_1_1[:, 1]
            p_2_2_0 = prob_2_2[:, 0]
            prob_class_4 = p_branch2 * p_2_2_0 * prob_3_3[:, 0]
            prob_class_5 = p_branch2 * p_2_2_0 * prob_3_3[:, 1]

            # Classes 6 & 7: branch "1-1" (output 1) then "2-2" (output 1) then "3-4"
            p_2_2_1 = prob_2_2[:, 1]
            prob_class_6 = p_branch2 * p_2_2_1 * prob_3_4[:, 0]
            prob_class_7 = p_branch2 * p_2_2_1 * prob_3_4[:, 1]

            # Stack probabilities into a [batch_size, 8] tensor.
            all_probs = torch.stack(
                [prob_class_0, prob_class_1, prob_class_2, prob_class_3,
                 prob_class_4, prob_class_5, prob_class_6, prob_class_7],
                dim=1
            )

            # Choose the final class with the highest aggregated probability.
            preds = torch.argmax(all_probs, dim=1)
            predictions.extend(preds.cpu().numpy())

    # Compute evaluation metrics.
    true_arr = np.array(true_classes)
    pred_arr = np.array(predictions)
    sq_distance = np.sum((true_arr - pred_arr) ** 2)
    rmse = np.sqrt(sq_distance / len(true_arr))
    accuracy = np.sum(true_arr == pred_arr) / len(true_arr)
    percentage_outside_rad_1 = np.sum(np.abs(true_arr - pred_arr) > 1) / len(true_arr)
    percentage_outside_rad_2 = np.sum(np.abs(true_arr - pred_arr) > 2) / len(true_arr)
    f1 = f1_score(torch.tensor(true_classes), torch.tensor(predictions), num_classes=8, task="multiclass")

    return true_arr, pred_arr, sq_distance, accuracy, rmse, percentage_outside_rad_1, percentage_outside_rad_2, f1


