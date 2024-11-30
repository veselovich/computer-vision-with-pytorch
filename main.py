import os
import torch
from src.models.create_model import create_effnetb0
from src.training.train import train, test_step
from src.models.save_model import save_model
from src.eval.visualize_results import pred_and_plot_image, plot_confusion_matrix_step, top_k_fails
from src.utils.writer import create_writer
from torchmetrics import Precision, Recall, F1Score
from torchvision import datasets
from torch.utils.data import DataLoader
from torchinfo import summary


def main():
    # Device agnostic code
    if torch.backends.mps.is_built():  # For macOS MPS
        device = torch.device("mps")
    elif torch.cuda.is_available():  # For NVIDIA GPUs
        device = torch.device("cuda")
    else:  # Fallback to CPU
        device = torch.device("cpu")

    device = "cpu"  # Works for MacBook Pro late 2013
    torch.set_default_device(device)

    # Setting up data
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Create a model and metrics
    model, model_transform = create_effnetb0(out_features=num_classes, device=device)

    # summary(
    #     model,
    #     input_size=(1, 3, 224, 224),
    #     # col_names=["input_size"], # uncomment for smaller output
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    #     )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Create DataLoaders
    num_workers = min(os.cpu_count(), 2)  # Works for MacBook Pro late 2013
    train_dataset.transform = model_transform
    test_dataset.transform = model_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    EPOCHS = 5

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

    writer = create_writer(
        data_name=train_dataset.__class__.__name__,
        model_name=model.name,
        extra=f"{EPOCHS}_epochs",
    )

    train_metrics = train(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
        writer=writer,
    )

    print(train_metrics)

    saved_model_path = save_model(model=model,
                                     traget_dir="models",
                                     model_name=f"{train_dataset.__class__.__name__}_{model.name}_{EPOCHS}_epochs.pth",)

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

    pred_and_plot_image(model=model,
                        class_names=class_names,
                        image_source="https://upload.wikimedia.org/wikipedia/commons/3/36/United_Airlines_Boeing_777-200_Meulemans.jpg",
                        transform=model_transform,
                        device=device,
                        )

    plot_confusion_matrix_step(model=model,
                               dataloader=test_loader,
                               device=device,
                               num_classes=num_classes,
                               class_names=class_names,
                               )
    
    top_k_fails(model=model,
                dataloader=test_loader,
                device=device,
                k=10)

if __name__ == "__main__":
    main()
