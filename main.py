import os
import torch
from itertools import product
from src.model import create_vision_model, get_vision_weights, save_model
from src.train import train, create_writer
from src.utils import get_device
from src.data import dataset_rand_reduce, load_dataset
from torch.utils.data import DataLoader


def main():
    device = get_device()
    print(f"Using device: {device}")

    EPOCHS_NUM = [10]
    LEARNING_RATE_VALS = [0.001]
    DATASETS = ["Flowers102"]
    REDUCE_DATASET_VALS = [1000]
    MODEL_NAMES = ["efficientnet_b0", "resnet18", "vit_b_16"]
    BATCH_SIZES = [32]
    COMPILE_MODEL_VALS = [False]

    param_combinations = product(
        EPOCHS_NUM, LEARNING_RATE_VALS, DATASETS, REDUCE_DATASET_VALS,
        MODEL_NAMES, BATCH_SIZES, COMPILE_MODEL_VALS
    )

    for EPOCHS, LEARNING_RATE, DATASET, REDUCE_DATASET, MODEL_NAME, BATCH_SIZE, COMPILE_MODEL in param_combinations:

        weights = get_vision_weights(MODEL_NAME)

        train_dataset = load_dataset(dataset_name=DATASET,
                                     transform=weights.transforms(),
                                     train=True)
        
        test_dataset = load_dataset(dataset_name=DATASET,
                                     transform=weights.transforms(),
                                     train=False)

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
            data_name=DATASET,
            model_name=MODEL_NAME,
            extra=f"{EPOCHS}_ep",
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
            model_name=f"{DATASET}_{MODEL_NAME}_{EPOCHS}_epochs.pth",
        )

if __name__ == "__main__":
    main()
