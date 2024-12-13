import os
import torch
from src.model import create_vision_model, get_vision_weights
from src.train import train, create_writer
from src.utils import get_device
from src.data import save_to_hdf5, dataset_rand_reduce
from torchvision import datasets
from torch.utils.data import DataLoader


def main():
    device = get_device()
    print(f"Using device: {device}")

    model_name = "efficientnet_b0"
    weights = get_vision_weights(model_name)

    # Setting up data
    train_dataset = datasets.CIFAR10(
        root="./data", transform=weights.transforms(), train=True, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="./data", transform=weights.transforms(), train=False, download=True
    )
    class_names = train_dataset.classes
    num_classes = len(class_names)

    REDUCE = 0.005
    train_subset = dataset_rand_reduce(train_dataset, reduce=REDUCE)
    test_subset = dataset_rand_reduce(test_dataset, reduce=REDUCE)

    # Create a model and metrics
    model, model_transform = create_vision_model(
        model_name=model_name,
        weights=weights,
        out_features=num_classes,
        device=device,
        print_summary=False,
        compile=False,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Create DataLoaders
    num_workers = os.cpu_count()
    pin_memory = device.type == "cuda"

    train_subset_h5 = save_to_hdf5(
        dataset=train_subset, save_path="./data/train_preprocessed.h5"
    )
    test_subset_h5 = save_to_hdf5(
        dataset=test_subset, save_path="./data/test_preprocessed.h5"
    )

    train_loader = DataLoader(
        train_subset_h5,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset_h5,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    EPOCHS = 1

    writer = create_writer(
        data_name=train_dataset.__class__.__name__,
        model_name=model.name,
        extra=f"{EPOCHS}_epochs",
    )

    train_metrics = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
        writer=writer,
    )

    print(train_metrics)


if __name__ == "__main__":
    main()
