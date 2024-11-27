import torch
import h5py
from torchvision import datasets, transforms
from tqdm import tqdm

# Preprocess and save as HDF5
def save_to_hdf5(dataset, save_path, transform):
    with h5py.File(save_path, 'w') as h5_file:
        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
            transformed_image = transform(image)
            h5_file.create_dataset(f'image_{idx}', data=transformed_image.numpy(), compression="gzip")
            h5_file.create_dataset(f'label_{idx}', data=label)
    print(f"Saved to {save_path}.")

# Load from HDF5
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = None  # To be opened lazily
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len([key for key in f.keys() if key.startswith("image_")])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
        image = torch.tensor(self.h5_file[f'image_{idx}'][:])
        label = int(self.h5_file[f'label_{idx}'][()])
        return image, label

# Example usage
if __name__ == "__main__":
    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    hdf5_path = "./data/preprocessed.h5"

    # Save dataset to HDF5
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    save_to_hdf5(raw_dataset, hdf5_path, transform)

    # Load from HDF5
    preprocessed_dataset = HDF5Dataset(hdf5_path)
    print(f"Loaded HDF5 dataset with {len(preprocessed_dataset)} samples.")
