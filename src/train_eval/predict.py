import torch
from torch_geometric.data import Data

from models.GAT import GAT


def predict_one_ensemble(ensemble: dict[str, GAT], data: Data, device = 'cpu') -> float:
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
