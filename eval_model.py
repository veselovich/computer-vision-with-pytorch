import os
import torch
from src.model import get_vision_weights, create_vision_model
from src.train import train, test_step, create_writer
from src.eval import pred_and_plot_image, plot_confusion_matrix_step, top_k_fails
from src.data import dataset_rand_reduce
from src.utils import get_device
from torchmetrics import Precision, Recall, F1Score
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

    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    EPOCHS = 1

    writer = create_writer(
        data_name=train_dataset.__class__.__name__,
        model_name=model.name,
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

    # Initialize additional metrics for model evaluation
    precision = Precision(
        num_classes=num_classes, task="multiclass", average="weighted"
    ).to(device)
    recall = Recall(num_classes=num_classes, task="multiclass", average="weighted").to(
        device
    )
    f1 = F1Score(num_classes=num_classes, task="multiclass", average="weighted").to(
        device
    )

    # Evaluate model
    test_metrics = test_step(
        model,
        test_loader,
        loss_fn,
        device,
        precision_metric=precision,
        recall_metric=recall,
        f1_metric=f1,
    )

    print(test_metrics)

    pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_source="https://upload.wikimedia.org/wikipedia/commons/3/36/United_Airlines_Boeing_777-200_Meulemans.jpg",
        transform=model_transform,
        device=device,
    )

    plot_confusion_matrix_step(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    top_k_fails(
        model=model, dataloader=test_loader, device=device, class_names=class_names, k=3
    )


if __name__ == "__main__":
    main()
