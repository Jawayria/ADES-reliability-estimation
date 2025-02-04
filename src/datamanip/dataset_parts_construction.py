import numpy as np
import pandas as pd
import torch

from decimal import Decimal, getcontext

from jupyterlab.utils import deprecated

from datamanip.feature_extraction import extract_features_from_data


def convert_range_to_floats(nine_range: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Converts a tuple of integer values representing decimal places into float values.
    Each integer represents the number of nines in a decimal fraction.

    Example: (1, 2, 3) -> (0.9, 0.99, 0.999)
    """
    return float("0." + "9" * nine_range[0]), float("0." + "9" * nine_range[1]), float("0." + "9" * nine_range[2])


def trim_df_by_range(df: pd.DataFrame, bottom_range: float, top_range: float) -> pd.DataFrame:
    """
    Filters the dataframe to include only rows where the 'reliability' column falls within a specified range.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'reliability' column.
        bottom_range (float): The minimum threshold for reliability values (inclusive).
        top_range (float): The maximum threshold for reliability values (exclusive).

    Returns:
        pd.DataFrame: A dataframe containing only rows with reliability values in the specified range.
    """
    return df.loc[(df['reliability'] >= bottom_range) & (df['reliability'] < top_range)]


def construct_node_features(df: pd.DataFrame) -> torch.Tensor:
    node_features_tensor = torch.tensor(extract_features_from_data(df), dtype=torch.float)
    print(node_features_tensor.shape)
    return node_features_tensor


def construct_edge_indices(df: pd.DataFrame) -> list[torch.Tensor]:
    # %%
    # Convert the matrices to edge indices

    all_edge_indices: list[torch.Tensor] = []

    df['matrix'] = df['matrix'].apply(np.array)

    for _, row in df.iterrows():
        matrix_np = row['matrix']

        # Get the indices where there are edges (i.e., non-zero entries)
        edge_indices = np.nonzero(matrix_np)

        # Stack the indices into a 2xN array where each column represents an edge
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)

        # Append the edge_index tensor to the list
        all_edge_indices.append(edge_index)
    return all_edge_indices



def construct_binary_classes(df: pd.DataFrame, threshold : float) -> torch.Tensor:
    all_rels_binary: list[int] = []

    # Iterate through the filtered dataframe and assign binary class labels
    for _, row in df.iterrows():
        reliability = row['reliability']
        if reliability >= threshold:
            all_rels_binary.append(1)  # Assign class 1 if reliability is above or equal to the threshold
        else:
            all_rels_binary.append(0)  # Assign class 0 otherwise

    # Convert the list of binary labels to a PyTorch tensor
    all_rels_tensor_binary: torch.Tensor = torch.tensor(all_rels_binary)
    return all_rels_tensor_binary


@deprecated
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

    # Do the actual binning
    all_rels: list[int] = []

    for _, row in merged_df_exploded.iterrows():
        reliability = row['reliability']
        bin_index = classify_into_bins(reliability, bins)
        all_rels.append(bin_index)
    return torch.tensor(all_rels)


def construct_reliability_classes(merged_df_exploded: pd.DataFrame, threshold : float,
                                  binary: bool = True) -> torch.Tensor:
    if binary:
        return construct_binary_classes(merged_df_exploded, threshold)
    else:
        raise NotImplementedError("Multiclass classification is not implemented for ladder datasets yet")
