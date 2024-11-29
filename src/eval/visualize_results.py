"""
Utility functions to make predictions.
"""

import requests
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_source: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = torch.device("cpu"),
):
    """Predicts on a target image (from a file path or URL) with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_source (str): Filepath or URL of the target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to "cpu".

    Returns:
        dict: Information about the image and prediction, including:
            - "predicted_label": Predicted class label
            - "predicted_probability": Predicted probability for the class
            - "image_shape": Shape of the original image
            - "transformed_image": The transformed image tensor
    """
    # Load image from URL or file path
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_source)

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

    # Prepare model for prediction
    model.to(device)
    model.eval()

    # Transform and add batch dimension to image
    transformed_image = image_transform(img).unsqueeze(dim=0).to(device)

    # Make prediction
    with torch.inference_mode():
        target_image_pred = model(transformed_image)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1).item()

    # Get prediction results
    predicted_label = class_names[target_image_pred_label]
    predicted_probability = target_image_pred_probs.max().item()

    # Plot image with prediction
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {predicted_label} | Prob: {predicted_probability:.3f}")
    plt.axis(False)
    plt.show()

    # Return image information and prediction
    return {
        "predicted_label": predicted_label,
        "predicted_probability": predicted_probability,
        "image_shape": img.size,  # (width, height)
        "transformed_image": transformed_image,
    }


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
    plot_confusion_matrix(
        conf_mat=conf_matrix, class_names=class_names, cmap=cmap, ax=ax
    )
    plt.title("Confusion Matrix")
    plt.show()
