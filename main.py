import os
import torch
from src.model import create_vision_model, get_vision_weights, save_model
from src.train import train, create_writer
from src.utils import get_device
from src.data import save_to_hdf5, dataset_rand_reduce
from torchvision import datasets
from torch.utils.data import DataLoader


def main():
    device = get_device()
    print(f"Using device: {device}")

    MODEL_NAME = "efficientnet_b0"
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    REDUCE_DATASET = 300
    COMPILE_MODEL = False

    weights = get_vision_weights(MODEL_NAME)

    # Setting up data
    train_dataset = datasets.CIFAR10(
        root="./data", transform=weights.transforms(), train=True, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="./data", transform=weights.transforms(), train=False, download=True
    )
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    train_subset = dataset_rand_reduce(train_dataset, num_samples=REDUCE_DATASET)
    test_subset = dataset_rand_reduce(test_dataset, num_samples=REDUCE_DATASET//5)

    # Create a model and metrics
    model, model_transform = create_vision_model(
        model_name=MODEL_NAME,
        weights=weights,
        out_features=num_classes,
        device=device,
        print_summary=False,
        compile=COMPILE_MODEL,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Optimizing data storage
    # train_subset = save_to_hdf5(
    #     dataset=train_subset, save_path="./data/train_preprocessed.h5"
    # )
    # test_subset = save_to_hdf5(
    #     dataset=test_subset, save_path="./data/test_preprocessed.h5"
    # )

    # Create DataLoaders
    num_workers = os.cpu_count()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    writer = create_writer(
        data_name=train_dataset.__class__.__name__,
        model_name=model.name,
        extra=f"{EPOCHS}_ep_hdf5",
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

    saved_model_path = save_model(
        model=model,
        target_dir="models",
        model_name=f"{train_dataset.__class__.__name__}_{model.name}_{EPOCHS}_epochs.pth",
    )

if __name__ == "__main__":
    main()
