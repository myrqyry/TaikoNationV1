import torch
from torch.quantization import quantize_dynamic

def apply_dynamic_quantization(model):
    """
    Applies dynamic INT8 quantization to compatible layers of the given model.

    This version selectively quantizes only the top-level Linear layers
    to avoid conflicts with the complex internal structure of modules like
    nn.TransformerEncoder, which can cause errors with quantized sub-modules.
    This provides a good balance of performance and stability.

    Args:
        model (torch.nn.Module): The model to be quantized.

    Returns:
        torch.nn.Module: The model with compatible layers quantized.
    """
    print("Applying INT8 dynamic quantization to compatible layers...")

    # We iterate through the direct children of the model.
    # This avoids recursing into complex modules like nn.TransformerEncoder.
    for module_name, module in model.named_children():
        # If a direct child module is a Linear layer, we quantize it.
        if isinstance(module, torch.nn.Linear):
            print(f"Quantizing layer: {module_name}")
            quantized_module = quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
            # Replace the original module with the new quantized version
            setattr(model, module_name, quantized_module)

    print("Selective quantization complete.")
    return model
