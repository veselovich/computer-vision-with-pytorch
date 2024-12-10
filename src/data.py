"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data and managing datasets stored in HDF5 format.
"""

import os
from typing import Tuple, List

import torch
import h5py
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

NUM_WORKERS = os.cpu_count()

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

def save_to_hdf5(dataset: Dataset, save_path: str, transform: transforms.Compose) -> None:
    """Saves a dataset to an HDF5 file after applying transformations.

    Args:
        dataset: PyTorch Dataset to save.
        save_path: Path to save the HDF5 file.
        transform: Transformation to apply to the dataset images.
    """
    with h5py.File(save_path, "w") as h5_file:
        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
            transformed_image = transform(image)
            h5_file.create_dataset(
                f"image_{idx}", data=transformed_image.numpy(), compression="gzip"
            )
            h5_file.create_dataset(f"label_{idx}", data=label)
    print(f"Dataset saved to {save_path}.")

class HDF5Dataset(Dataset):
    """Dataset for loading data stored in HDF5 format."""

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.h5_file = None  # Lazy loading
        with h5py.File(hdf5_path, "r") as f:
            self.length = len([key for key in f.keys() if key.startswith("image_")])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")
        image = torch.tensor(self.h5_file[f"image_{idx}"][:])
        label = int(self.h5_file[f"label_{idx}"][()])
        return image, label

if __name__ == "__main__":
    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    hdf5_path = "./data/preprocessed.h5"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    save_to_hdf5(raw_dataset, hdf5_path, transform)

    preprocessed_dataset = HDF5Dataset(hdf5_path)
    print(f"Loaded HDF5 dataset with {len(preprocessed_dataset)} samples.")