from decimal import Decimal
from itertools import permutations
import pandas as pd

from datamanip.read_csvs import read_matrices, read_rel_values, merge_matrices_and_rel
from filepath import matrices_path, reliabilities_path, config_all_path


class Segment:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def contains(self, value: int) -> bool:
        return self.start < value < self.end


def find_max_num_of_nines(df: pd.DataFrame) -> int:
    """
    Find the maximum number of nines in the 'reliability' column of a dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'reliability' column.

    Returns:
        int: The maximum number of nines in the 'reliability' column.
    """
    # Find the biggest reliability value that is less than 1
    max_reliability = df.loc[df['reliability'] < 1, 'reliability'].max()

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
    return count_nines


def convert_permutation_to_tournament(permutation: tuple[int, ...], segments: list[Segment]) -> list[tuple[int, int, int]]:
    tournament = []
    for place_to_cut in permutation:
        for segment in segments:
            if segment.contains(place_to_cut):
                tournament.append((segment.start, place_to_cut, segment.end))
                segments.remove(segment)
                segments.append(Segment(segment.start, place_to_cut))
                segments.append(Segment(place_to_cut, segment.end))
                break
    return tournament

#def is_valid_tournament(tournament: list[tuple[int, int, int]], num_of_clases : int) -> bool:

def count_classes(df: pd.DataFrame) -> dict[int, int]:
    """
    Categorizes floating-point reliability values into discrete classes based on the number of consecutive nines at the beginning.

    Args:
        df (pd.DataFrame): DataFrame containing a 'reliability' column with floating-point values.

    Returns:
        dict[int, int]: A dictionary mapping class IDs (based on leading nines count) to their occurrences.
    """

    def count_leading_nines(value: float) -> int:
        str_val = f"{value:.16f}"  # Convert float to string
        if not str_val.startswith("0.9"):
            return 0  # If it doesn't start with 0.9, it's class 0
        return len(str_val.split("9")[0]) - 2  # Count consecutive nines after "0."

    class_counts = df['reliability'].apply(count_leading_nines).value_counts().to_dict()
    return class_counts


def evaluate_class_imbalance(tournament: list[tuple[int, int, int]], class_counts: dict[int, int]) -> float:
    """
    Evaluates class balance across all matches in a tournament using a DataFrame.

    Args:
        tournament (list[tuple[int, int, int]]): List of tuples representing matches (start_class, threshold, end_class).
        df (pd.DataFrame): DataFrame containing a 'reliability' column with floating-point values.

    Returns:
        float: A score representing the overall class balance (lower is better).
    """

    imbalance_score = 0.0

    for start_class, threshold, end_class in tournament:
        # Get counts for the two groups being compared
        class1_count = sum(class_counts.get(i, 1) for i in range(start_class, threshold))
        class2_count = sum(class_counts.get(i, 1) for i in range(threshold, end_class))

        # Avoid division by zero
        if class1_count == 0 or class2_count == 0:
            continue

        # Compute imbalance ratio and apply quadratic penalty
        ratio = max(class1_count, class2_count) / min(class1_count, class2_count)
        penalty = ratio ** 2  # Quadratic penalty to emphasize severe imbalances
        imbalance_score += penalty

    return imbalance_score


def main():
    all_matrices_df = read_matrices(matrices_path)
    all_rels_df = read_rel_values(reliabilities_path, config_all_path)
    merged_df = merge_matrices_and_rel(all_matrices_df, all_rels_df)
    exploded_df = merged_df.explode(['timestamp', 'reliability']).reset_index(drop=True)
    # Remove all rows where timestamp is above 10000
    df = exploded_df[exploded_df['timestamp'] <= 10000]

    max_num_of_nines = find_max_num_of_nines(df)
    numbers = list(range(max_num_of_nines + 1))[1:]
    all_permutations = list(permutations(numbers))
    all_tournaments = []
    for permutation in all_permutations:
        segments: list[Segment] = [Segment(0, max_num_of_nines + 1)]
        ranges = convert_permutation_to_tournament(permutation, segments)
        all_tournaments.append(ranges)

    class_counts = count_classes(df)
    best_tournament = min(all_tournaments, key=lambda t: evaluate_class_imbalance(t, class_counts))
    print(best_tournament)

if __name__ == '__main__':
    main()
