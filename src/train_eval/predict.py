import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data

from datamanip.datasetmanip.feature_extraction import get_meta_features_from_ensemble


def predict_one_ensemble(ensemble: dict[str, torch.nn.Module], data: Data, device='cpu') -> float:
    """
    Predicts the winner of a match based on the ensemble of models.

    Args:
        ensemble (dict): A dictionary containing the best model for each match.
        data (Data): A PyTorch Geometric Data object containing the input features.

    Returns:
        int: The predicted reliability class.
    """
    for match, model in ensemble.items():
        model.eval()

    with torch.no_grad():
        data = data.to(device)
        match = "1-1"
        out = ensemble[match](data)
        pred = out.argmax(dim=1)  # Get predicted class
        if pred == 0:
            match = "2-1"
            out = ensemble[match](data)
            pred = out.argmax(dim=1)
            if pred == 0:
                match = "3-1"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    return 0
                else:
                    return 1
            else:
                match = "3-2"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    return 2
                else:
                    return 3
        else:
            match = "2-2"
            out = ensemble[match](data)
            pred = out.argmax(dim=1)
            if pred == 0:
                match = "3-3"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    return 4
                else:
                    return 5
            else:
                match = "3-4"
                out = ensemble[match](data)
                pred = out.argmax(dim=1)
                if pred == 0:
                    return 6
                else:
                    return 7


def predict_one_meta_learner(ensemble: dict[str, torch.nn.Module], meta_learner: RandomForestClassifier, data: Data,
                             device='cpu') -> float:
    """
    Predicts the winner of a match based on the meta-learner model.

    Args:
        meta_learner (RandomForestClassifier): The trained meta-learner model.
        data (Data): A PyTorch Geometric Data object containing the input features.

    Returns:
        int: The predicted reliability class.
    """
    outputs = get_meta_features_from_ensemble(ensemble, data, device)
    prediction = meta_learner.predict(outputs)[0]
    return prediction
