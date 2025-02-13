# Import necessary libraries
import numpy as np
from time import perf_counter

import torch
from torch_geometric.loader import DataLoader


from datamanip.datasetmanip.three_five_dataset import ThreeFiveDataset
from filepath import dataset_path
from train_eval.predict import predict_one_ensemble
from utils import load_best_model_based_on_match


def calc_inference_times():
    # Convert dataset to PyTorch tensors
    dataset = ThreeFiveDataset(dataset_path, 'ensemble', "test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    NODE_FEATURES = 12
    DROPOUT_RATE = 0.3
    # Measure inference times
    inference_times = []
    total_time = 0

    matches = ["1-1", "2-1", "2-2", "3-1", "3-2", "3-3", "3-4"]
    ensemble = {}
    for match in matches:
        ensemble[match] = load_best_model_based_on_match(match)

    with torch.no_grad():
        for data in dataloader:
            start_time = perf_counter()
            _ = predict_one_ensemble(ensemble, data)
            end_time = perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            total_time += inference_time

    # Sort by inference time
    sorted_times = sorted(inference_times)

    # Extract results
    best_time = sorted_times[0]
    worst_time = sorted_times[-1]
    average_time = np.mean(sorted_times)

    # Print results
    print("Best Inference Time:", best_time)
    print("Worst Inference Time:", worst_time)
    print("Average Inference Time:", average_time)
    print("Total Inference Time:", total_time)


if __name__ == "__main__":
    print("Calculating inference times...")
    calc_inference_times()