
import numpy as np
import pandas as pd
import torch

from decimal import Decimal, getcontext



from datamanip.feature_extraction import extract_features_from_data


def construct_node_features(merged_df_exploded: pd.DataFrame, num_features: int = 5) -> torch.Tensor:

    node_features_tensor = torch.tensor(extract_features_from_data(merged_df_exploded), dtype=torch.float)
    return node_features_tensor

def construct_edge_indices(merged_df_exploded: pd.DataFrame) -> list[torch.Tensor]:
    # %%
    # Convert the matrices to edge indices

    all_edge_indices: list[torch.Tensor] = []

    merged_df_exploded['matrix'] = merged_df_exploded['matrix'].apply(np.array)

    for _, row in merged_df_exploded.iterrows():
        matrix_np = row['matrix']

        # Get the indices where there are edges (i.e., non-zero entries)
        edge_indices = np.nonzero(matrix_np)

        # Stack the indices into a 2xN array where each column represents an edge
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)

        # Append the edge_index tensor to the list
        all_edge_indices.append(edge_index)
    return all_edge_indices


# Convert the reliability values to classes (two classes: 0 and 1)
def construct_binary_classes(merged_df_exploded: pd.DataFrame, threshold: float = 0.99) -> torch.Tensor:
    all_rels_binary: list[int] = []

    for _, row in merged_df_exploded.iterrows():
        reliability = row['reliability']
        if reliability >= threshold:
            all_rels_binary.append(1)
        else:
            all_rels_binary.append(0)
    # Assumes that classes in the list are integers (0 or 1 for binary case, 0,1,2,3,4.... for multiclass case)
    all_rels_tensor_binary: torch.Tensor = torch.tensor(all_rels_binary)
    return all_rels_tensor_binary


def construct_multiclass_classes(merged_df_exploded: pd.DataFrame) -> torch.Tensor:
    # Find the biggest reliability value that is less than 1
    max_reliability = merged_df_exploded.loc[merged_df_exploded['reliability'] < 1, 'reliability'].max()

    # Take the part after the decimal point
    max_value_decimal = Decimal(str(max_reliability))  # Ensure precise representation
    decimal_part = str(max_value_decimal).split('.')[1]  # Get the decimal part

    # Count the number of nines in the decimal part
    count_nines = 0
    for digit in decimal_part:
        if digit == '9':
            count_nines += 1
        else:
            break  # Stop counting when encountering the first non-9 digit

    getcontext().prec = count_nines + 2  # Set precision
    # Create bins
    bins = []
    current_bin_start = Decimal('0')

    for i in range(1, count_nines + 1):
        current_bin_end = Decimal('1') - Decimal(f'1e-{i}')
        bins.append((current_bin_start, current_bin_end))
        current_bin_start = current_bin_end
    # Define function to put values into bins
    def classify_into_bins(value, bins_nines):
        for idx, (bin_start, bin_end) in enumerate(bins_nines):
            if bin_start <= value < bin_end:
                return idx
        return len(bins)

    #Do the actual binning
    all_rels: list[int] = []

    for _, row in merged_df_exploded.iterrows():
        reliability = row['reliability']
        bin_index = classify_into_bins(reliability, bins)
        all_rels.append(bin_index)
    return torch.tensor(all_rels)


def construct_reliability_classes(merged_df_exploded: pd.DataFrame, threshold: float = 0.99) -> (
        torch.Tensor, torch.Tensor):
    return (construct_binary_classes(merged_df_exploded, threshold), construct_multiclass_classes(merged_df_exploded))
