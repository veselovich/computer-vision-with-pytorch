import torch

# Detect the best available device
if torch.has_mps:  # For macOS MPS
    device = torch.device("mps")
elif torch.cuda.is_available():  # For NVIDIA GPUs
    device = torch.device("cuda")
else:  # Fallback to CPU
    device = torch.device("cpu")

print(f"Using device: {device}")