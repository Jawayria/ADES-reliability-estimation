import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def split_dataset(data_list: pd.DataFrame, test_size=0.2, val_size=0.25, random_state=42) -> tuple[
   pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a list of Data objects into training, testing, and validation sets.

    """
    # Split into training and testing data
    train_data_list, test_data_list = train_test_split(data_list, test_size=test_size, random_state=random_state)

    # Further split training data into training and validation
    train_data_list, val_data_list = train_test_split(train_data_list, test_size=val_size,
                                                      random_state=random_state)  # 0.25 x 0.8 = 0.2 validation split
    return train_data_list, test_data_list, val_data_list

def create_loaders(train_data_list, val_data_list, test_data_list):
    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
