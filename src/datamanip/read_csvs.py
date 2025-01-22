import glob
import pandas as pd

from utils import extract_config_id


def read_matrices(filepath: str) -> pd.DataFrame:
    """
    Read all matrix files and convert them into a list of lists
    """
    mat_files = sorted(glob.glob(filepath))

    all_matrices = []

    for path in mat_files:

        with open(path, 'r') as f:
            content = f.read()

        # Remove the first and last character ('[' and ']')
        content = content[1:-1]
        string_list = []
        current_row = ''
        start_line = False

        for c in content:
            if c == '[':
                start_line = True
            elif c == ']':
                start_line = False
                string_list.append(current_row)
                current_row = ''
            elif start_line:
                if c != '\n':
                    current_row += c
        # Convert the string representation of the matrix into a list of lists
        matrix = []
        for string in string_list:
            vals = string.replace('.', ' ').split()
            vals = [int(v) for v in vals]
            matrix.append(vals)
        all_matrices.append({"matrix": matrix, "config_id": extract_config_id(path)})

    # Create DataFrame for future use
    all_matrices_df = pd.DataFrame(all_matrices)
    return all_matrices_df


def read_rel_values(filepath: str, path_to_config_all: str) -> pd.DataFrame:
    """
    Read all reliability files and convert them into a list of lists
    """
    rel_files = sorted(glob.glob(filepath))
    all_rels = []
    if path_to_config_all in rel_files:
        rel_files.remove(path_to_config_all)

    for path in rel_files:
        rel_data = pd.read_csv(
            path,
            sep=';',
            decimal=',',
            names=['timestamp', 'reliability']
        )

        all_rels.append({"rel_data": rel_data, "config_id": extract_config_id(path)})
    all_rels_df = pd.DataFrame(all_rels)
    return all_rels_df


def merge_matrices_and_rel(all_matrices_df: pd.DataFrame, all_rels_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(all_matrices_df, all_rels_df, on="config_id", how="inner")

    # We assume merged_df has a 'rel_data' column that is a DataFrame
    # with columns 'timestamp' and 'reliability'

    # Copy the columns to the parent DataFrame
    merged_df['timestamp'] = merged_df['rel_data'].apply(lambda df: df['timestamp'].tolist())
    merged_df['reliability'] = merged_df['rel_data'].apply(lambda df: df['reliability'].tolist())

    # Drop the child dataframe
    merged_df = merged_df.drop(columns=['rel_data'])

    return merged_df
