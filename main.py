import os
import torch
from src.models.create_model import create_effnetb0
from src.training.train import test_step
from torchmetrics import Precision, Recall, F1Score
from torchvision import datasets
from torch.utils.data import DataLoader

def main():
    # Device agnostic code
    if torch.backends.mps.is_built():  # For macOS MPS
        device = torch.device("mps")
    elif torch.cuda.is_available():  # For NVIDIA GPUs
        device = torch.device("cuda")
    else:  # Fallback to CPU
        device = torch.device("cpu")
    
    device = "cpu" # Works for MacBook Pro late 2013
    torch.set_default_device(device)

    # Setting up data
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Create a model and metrics
    model, model_transform = create_effnetb0(out_features=num_classes,
                        device=device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)
    
    # Initialize additional metrics
    precision = Precision(num_classes=num_classes, task='multiclass', average='weighted').to(device)
    recall = Recall(num_classes=num_classes, task='multiclass', average='weighted').to(device)
    f1 = F1Score(num_classes=num_classes, task='multiclass', average='weighted').to(device)

    # Create DataLoaders
    num_workers = min(os.cpu_count(), 2) # Works for MacBook Pro late 2013
    train_dataset.transform = model_transform
    test_dataset.transform = model_transform

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)


    # Test model
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

if __name__ == "__main__":
    main()