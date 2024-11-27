"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)


def plot_confusion_matrix_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[list] = None,
    figsize: tuple = (8, 8),
    cmap: str = "Blues",
) -> None:
    """Computes and plots the confusion matrix for a PyTorch model using mlxtend.

    Args:
    model: A PyTorch model to be evaluated.
    dataloader: A DataLoader instance for the test dataset.
    device: A target device to compute on (e.g., "cuda" or "cpu").
    num_classes: The number of classes in the dataset.
    class_names: Optional list of class names for the confusion matrix labels.
    figsize: Tuple specifying the size of the confusion matrix plot.
    cmap: Colormap for the confusion matrix visualization.

    Returns:
    None. Displays the confusion matrix plot.
    """
    # Put model in eval mode
    model.eval()

    # Initialize confusion matrix metric
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes).to(device)

    # Turn on inference mode
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Get predictions
            test_pred_labels = test_pred_logits.argmax(dim=1)

            # 3. Update confusion matrix metric
            confusion_matrix_metric.update(test_pred_labels, y)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix_metric.compute().cpu().numpy()

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    plot_confusion_matrix(conf_mat=conf_matrix, class_names=class_names, cmap=cmap, ax=ax)
    plt.title("Confusion Matrix")
    plt.show()