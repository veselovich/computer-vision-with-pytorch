import torch
from torch import nn
import torch.quantization as quantization
from torchvision import transforms

def quantize_model(model: torch.nn.Module,
                          quant_type: str = "dynamic",
                          transform: transforms.Compose = None,
                          device: torch.device = torch.device("cpu")):
    """
    Quantizes a vision model from PyTorch model zoo.

    Args:
        model_name (str): Name of the model (e.g., "resnet18", "mobilenet_v2").
        quantization_type (str): Type of quantization ("dynamic" or "static").
        device (str): Device to run quantization on (default is "cpu").

    Returns:
        quantized_model: The quantized model.
    """

    model.eval()  # Set the model to evaluation mode

    if quant_type == "dynamic":
        # Apply dynamic quantization
        quantized_model = quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8  # Quantization data type
        )

    elif quant_type == "static":
        # Prepare the model for static quantization
        model.qconfig = quantization.get_default_qconfig('x86')

        # Fuse layers if supported by the model architecture
        model = fuse_model_layers(model=model)

        # Prepare and calibrate the model
        quantization.prepare(model, inplace=True)

        # Calibration with dummy data
        dummy_input = torch.rand(1, 3, 224, 224).to(device)
        if transform:
            dummy_input = transform(dummy_input)
        model(dummy_input)

        # Convert to quantized version
        quantized_model = quantization.convert(model, inplace=True)

    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    return quantized_model


def fuse_model_layers(model):
    # Try to fuse layers in the model based on known patterns
    # This works for models like ResNet, VGG, etc.

    # For ResNet-like models, fuse common layers
    if isinstance(model, nn.Sequential):
        # Fusing Conv + BatchNorm + ReLU layers
        model = quantization.fuse_modules(model, [['0', '1', '2']])  # Example fusion
    elif isinstance(model, nn.Module):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Conv2d) and hasattr(model, name + '.bn'):
                # Check if there is a BatchNorm and fuse it with Conv
                model = quantization.fuse_modules(model, [[name, name + '.bn', name + '.relu']])

    return model