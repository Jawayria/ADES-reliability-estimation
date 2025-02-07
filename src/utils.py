import os
import re

import pandas as pd
import torch

from filepath import hyperparameters_path, best_models_path, model_checkpoints_path
from models.GAT import GAT


def extract_config_id(filename):
    match = re.search(r"config_(\d+)", filename)
    return int(match.group(1)) if match else None

def load_best_model_based_on_match(match : str) -> GAT:
    # Find the best hyperparameters for each match from the .csv
    best_hyperparameters_df = pd.read_csv(os.path.join(hyperparameters_path, f"{match}.csv"))
    # Extract first row values
    first_row = best_hyperparameters_df.iloc[0]

    # Assign extracted values to variables
    NODE_FEATURES = int(first_row["Node features"])
    HIDDEN_DIM = int(first_row["Hidden dimension"])
    DROPOUT_RATE = float(first_row["Dropout rate"])

    # Initialize the GAT model with extracted parameters
    model = GAT(input_dim=NODE_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=2, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(os.path.join(best_models_path, f"{match}.pth"), weights_only=True))
    return model


