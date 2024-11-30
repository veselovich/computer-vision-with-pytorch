import torch
from torch import nn
from typing import Type, Optional

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
    # Instantiate the model architecture
    model = model_class()
    
    # Load the state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)  # Move the model to the specified device
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model