"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data and managing datasets stored in HDF5 format.
"""

import os
from typing import Tuple, List, Optional

import torch
import h5py
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

NUM_WORKERS = os.cpu_count()


class HDF5Dataset(Dataset):
    """Dataset for loading data stored in HDF5 format."""

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self.image_keys = [key for key in f.keys() if key.startswith("image_")]
            self.label_keys = [key for key in f.keys() if key.startswith("label_")]
            self.length = len(self.image_keys)
        self.h5_file = None  # Lazy loading

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range.")
        image = torch.tensor(self.h5_file[self.image_keys[idx]][:])
        label = int(self.h5_file[self.label_keys[idx]][()])
        return image, label

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        self.h5_file = h5py.File(self.hdf5_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_to_hdf5(
    dataset: Dataset, save_path: str, transform: Optional[transforms.Compose] = None
) -> None:
    """Saves a dataset to an HDF5 file after optionally applying transformations.

    Args:
        dataset: PyTorch Dataset to save.
        save_path: Path to save the HDF5 file.
        transform: Optional transformation to apply to the dataset images.

    Returns:
        HDF5Dataset made from original dataset.
    """
    with h5py.File(save_path, "w") as h5_file:
        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
            if transform is not None:
                image = transform(image)
            h5_file.create_dataset(
                f"image_{idx}", data=image.numpy(), compression="gzip"
            )
            h5_file.create_dataset(f"label_{idx}", data=label)
    print(f"Dataset saved to {save_path}.")

    return HDF5Dataset(save_path)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders from directories.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to apply to the data.
        batch_size: Number of samples per batch in each DataLoader.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_dataloader, test_dataloader, class_names).
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


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
