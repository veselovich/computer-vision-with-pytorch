import torch
import torchvision
from torch import nn
from torchinfo import summary
from typing import Type, Optional
from pathlib import Path

def create_effnetb0(out_features: int, device: torch.device, seed: Optional[int] = None,
                    print_summary: bool = False, compile: bool = True):
    """
    Creates an EfficientNetB0 feature extractor model.

    Args:
        out_features (int): Number of output features for the classifier head.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        seed (Optional[int]): Seed for reproducibility.
        print_summary (bool): Whether to print the model summary.
        compile (bool): Whether to compile the model using Torch 2.0.

    Returns:
        model (torch.nn.Module): The created EfficientNetB0 model.
        transform (torchvision.transforms): Preprocessing transforms for the model.
    """
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    transform = weights.transforms()

    for param in model.features.parameters():
        param.requires_grad = False

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=out_features)
    ).to(device)

    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")

    if print_summary:
        summary(
            model,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )

    if compile:
        try:
            model = torch.compile(model)
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            model.eval()
            with torch.no_grad():
                model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print(f"[INFO] Model {model.name} is compiled.")
        except Exception as e:
            print(f"[WARNING] Error compiling model: {e}")

    return model, transform

def load_model(model_class: Type[nn.Module], model_path: str, device: str = 'cpu') -> Optional[nn.Module]:
    """
    Loads a PyTorch model from a file.

    Args:
        model_class (Type[nn.Module]): The class of the model to instantiate.
        model_path (str): The path to the saved model file.
        device (str): The device to map the model to ('cpu' or 'cuda').

    Returns:
        Optional[nn.Module]: The loaded PyTorch model, or None if loading fails.
    """
    model = model_class()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model

def save_model(model: nn.Module, target_dir: str, model_name: str) -> str:
    """
    Saves a PyTorch model to a target directory and returns the absolute path to the saved model.

    Args:
        model (nn.Module): A target PyTorch model to save.
        target_dir (str): Directory for saving the model.
        model_name (str): Filename for the saved model (must end with ".pth" or ".pt").

    Returns:
        str: Absolute path to the saved model file.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    return str(model_save_path.resolve())
