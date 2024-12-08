import torch
import torchvision
from torch import nn
from torchinfo import summary

# Create an EffNetB0 feature extractor
def create_effnetb0(out_features: int,
                    device: torch.device,
                    seed: int = None,
                    print_summary: bool = False,
                    compile: bool = True):
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

    if print_summary:
        summary(
            model,
            input_size=(1, 3, 224, 224),
            # col_names=["input_size"], # uncomment for smaller output
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