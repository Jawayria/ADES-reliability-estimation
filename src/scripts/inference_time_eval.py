# Import necessary libraries
import numpy as np
from time import perf_counter

import torch
from torch_geometric.loader import DataLoader


from datamanip.datasetmanip.three_five_dataset import ThreeFiveDataset
from filepath import dataset_path, model_checkpoints_path
from models.GAT import GAT

def calc_inference_times():
    # Convert dataset to PyTorch tensors
    dataset = ThreeFiveDataset(root=dataset_path, load_binary=True)
    filtered_data = [data for data in dataset if data.x[0][3] == 1000]
    print(f"Number of samples: {len(filtered_data)}")
    dataloader = DataLoader(filtered_data, batch_size=1, shuffle=False)
    NODE_FEATURES = 5
    DROPOUT_RATE = 0.3
    model =  GAT(input_dim=NODE_FEATURES, hidden_dim=64, output_dim=2, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(model_checkpoints_path + "best_model_GAT.pth", weights_only=True))
    # Measure inference times
    inference_times = []
    total_time = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            start_time = perf_counter()
            _ = model(data)
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
    calc_inference_times()