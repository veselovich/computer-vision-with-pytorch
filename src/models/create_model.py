import torch
import torchvision
from torch import nn

# Create an EffNetB0 feature extractor
def create_effnetb0(out_features: int,
                    device: torch.device,
                    seed: int = None,):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    transform = weights.transforms()

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    if seed:
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)

    in_features = model.classifier[1].in_features

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=out_features)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model, transform