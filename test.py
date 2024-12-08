import torch
import time
import os
from src.models.create_model import create_effnetb0
from src.models.quantization import quantize_model

def get_model_size(model):
    # Convert model to CPU and save temporarily to check size
    tmp_path = "tmp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    model_size = os.path.getsize(tmp_path) / (1024 * 1024)  # Size in MB
    os.remove(tmp_path)
    return model_size

def measure_inference_time(model, device, input_tensor, num_runs=100):
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warm-up phase: Run a few inferences before timing

    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)
    
    # Measure total inference time for multiple runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_tensor)
    end_time = time.time()
    
    # Calculate average inference time per run
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    return avg_time_per_run


def main():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create a model and metrics
    model, model_transform = create_effnetb0(out_features=10,
                                             device=device,
                                             print_summary=False,
                                             compile=False)

    # Measure model size before quantization
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")

    # Measure inference time before quantization
    dummy_input = torch.randn(1, 3, 224, 224)  # Assuming the model accepts this input size
    original_inference_time = measure_inference_time(model, device, dummy_input, num_runs=100)
    print(f"Original average inference time for 100 runs: {original_inference_time:.4f} seconds")

    # Quantize the model
    model = quantize_model(model=model,
                           quant_type="dynamic",
                           transform=model_transform,
                           device=device)
    print("model quantized")
    model = torch.compile(model, backend="eager")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Measure model size after quantization
    quantized_size = get_model_size(model)
    print(f"Quantized model size: {quantized_size:.2f} MB")

    # Measure inference time after quantization
    quantized_inference_time = measure_inference_time(model, device, dummy_input, num_runs=100)
    print(f"Quantized average inference time for 100 runs: {quantized_inference_time:.4f} seconds")

if __name__ == "__main__":
    main()
