import os
import torch
import numpy as np
from torch.utils.data import Subset


def dataset_rand_reduce(dataset, reduce: float = 0.2) -> Subset:
    """
    Reduces the dataset to a random subset of specified size.

    Args:
        dataset: The original dataset to reduce.
        reduce (float): Fraction of the dataset to retain. Defaults to 0.2.

    Returns:
        Subset: A random subset of the original dataset.
    """
    subset_size = int(len(dataset) * reduce)
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    return Subset(dataset, subset_indices)

def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculates the size of a model's state dictionary in megabytes (MB).

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        float: Size of the model in MB.
    """
    tmp_path = "tmp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    model_size = os.path.getsize(tmp_path) / (1024 * 1024)  # Convert bytes to MB
    os.remove(tmp_path)
    return model_size
