import torch
from torch import nn
from src.utils import get_model_size

import torch
from torch import nn
import torch.ao.quantization as quantization
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
            {torch.nn.Linear},
            dtype=torch.qint8
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


def main():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(26 * 26 * 16, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # Create an instance of the model
    model = SimpleModel()

    # Measure model size before quantization
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")


    # Quantize the model
    model = quantize_model(model=model,
                           quant_type="dynamic",
                           device=device)
    print("model quantized")

    model = torch.compile(model, backend="eager")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Measure model size after quantization
    quantized_size = get_model_size(model)
    print(f"Quantized model size: {quantized_size:.2f} MB")

    print(model)

    print(model(dummy_input))

if __name__ == "__main__":
    main()