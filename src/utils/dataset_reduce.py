import numpy as np
from torch.utils.data import Subset

def dataset_rand_reduce(dataset, reduce: float = 0.2):

    subset_size = int(len(dataset) * reduce)
    # Randomly select indices for the subset
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)

    return Subset(dataset, subset_indices)