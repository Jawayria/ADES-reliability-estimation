import os

import importlib


import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

import models
import pickle

importlib.reload(models)
from filepath import *
from models.GAT import GAT
from models.GAT_LN_HEAD import GAT_LN_HEAD


def load_best_model_based_on_match(match : str) -> torch.nn.Module:
    # Find the best hyperparameters for each match from the .csv
    best_hyperparameters_df = pd.read_csv(os.path.join(hyperparameters_path, f"{match}.csv"))
    # Extract first row values
    first_row = best_hyperparameters_df.iloc[0]

    # Assign extracted values to variables
    NODE_FEATURES = int(first_row["Node features"])
    HIDDEN_DIM = int(first_row["Hidden dimension"])
    DROPOUT_RATE = float(first_row["Dropout rate"])
    OUTPUT_DIM = int(first_row["Output dimension"])
    # Initialize the GAT model with extracted parameters
    if match == "1-1" or match == "2-1" or   match == "2-2" or match == "3-2" or match == "3-3" or match == "3-4" or match == "4-1" or match == "4-2" or match == "ensemble":
        heads = 4
        if match == "3-2":
            heads = 8
        print(f"Using GAT_LN_HEAD for match {match}, using {heads} heads")
        model = GAT_LN_HEAD(input_dim=NODE_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE, num_heads=heads)
        model.load_state_dict(torch.load(os.path.join(best_models_path, f"{match}.pth"), weights_only=True))
    else:
        print(f"Using GAT for match {match}")
        model = GAT(input_dim=NODE_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
        model.load_state_dict(torch.load(os.path.join(best_models_path, f"{match}.pth"), weights_only=True))


    return model


def load_best_ensemble(multiclass_included=False) -> dict[str, torch.nn.Module]:
    matches = ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]
    if multiclass_included: matches.append("ensemble")
    ensemble = {}
    for match in matches:
        ensemble[match] = load_best_model_based_on_match(match)
        ensemble[match].eval()
    return ensemble

def load_best_meta_learner() -> RandomForestClassifier:
    # Load the best forest model
    with open(os.path.join(meta_learner_model_path, 'forest_model.pkl'), 'rb') as f:
        forest_model = pickle.load(f)
    return forest_model

def load_best_solution() -> tuple[dict[str, torch.nn.Module], RandomForestClassifier]:
    return load_best_ensemble(), load_best_meta_learner()


