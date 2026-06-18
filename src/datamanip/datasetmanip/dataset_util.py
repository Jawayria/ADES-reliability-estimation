import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def split_dataset(data, test_size=0.2, val_size=0.25, random_state=42):
    """
    Works for both:
    - DataFrame
    - numpy arrays / lists (e.g., config_ids)
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=val_size, random_state=random_state)
    return train, test, val

def create_loaders(train_data_list, val_data_list, test_data_list):
    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
