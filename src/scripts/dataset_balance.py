from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datamanip.datasetmanip.three_five_dataset import ThreeFiveDataset
from filepath import dataset_path

def main():

    #Count class occurences for unlimited dataset
    dataset_path_full = str(Path(dataset_path).parent / '3-switches-5-slaves')
    train_set = ThreeFiveDataset(root=dataset_path_full, split="train", max_hours=100000)
    test_set = ThreeFiveDataset(root=dataset_path_full, split="test", max_hours=100000)
    val_set = ThreeFiveDataset(root=dataset_path_full, split="val", max_hours=100000)

    print("Imported original dataset, counting class occurences...")

    classes = []
    for data in train_set:
        classes.extend(data.y.cpu().numpy())
    for data in test_set:
        classes.extend(data.y.cpu().numpy())
    for data in val_set:
        classes.extend(data.y.cpu().numpy())
    class_occurences = Counter(classes)
    # Count class occurences for 10k hours
    dataset_path_10k = dataset_path
    train_set_10k = ThreeFiveDataset(root=dataset_path_10k, split="train", max_hours=10000)
    test_set_10k = ThreeFiveDataset(root=dataset_path_10k, split="test", max_hours=10000)
    val_set_10k = ThreeFiveDataset(root=dataset_path_10k, split="val", max_hours=10000)

    print("Imported reduced dataset, counting class occurences...")
    classes_10k = []
    for data in train_set_10k:
        classes_10k.extend(data.y.cpu().numpy())
    for data in test_set_10k:
        classes_10k.extend(data.y.cpu().numpy())
    for data in val_set_10k:
        classes_10k.extend(data.y.cpu().numpy())
    class_occurences_10k = Counter(classes_10k)
    #Plot class balance bar chart
    print("Plotting...")

    # Union of all classes
    classes = sorted(set(class_occurences.keys()) | set(class_occurences_10k.keys()))

    before = np.array([class_occurences.get(c, 0) for c in classes])
    after = np.array([class_occurences_10k.get(c, 0) for c in classes])

    removed = before - after

    plt.figure(figsize=(12, 6))

    # Removed samples (bottom)
    plt.bar(
        classes,
        removed,
        color="tab:orange",
        edgecolor="black",
        hatch="//",
        label="Removed during preprocessing",
    )

    # Remaining samples (top)
    plt.bar(
        classes,
        after,
        bottom=removed,
        color="forestgreen",
        edgecolor="black",
        label="After preprocessing",
    )

    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class balance before and after preprocessing")

    plt.xticks(classes)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()