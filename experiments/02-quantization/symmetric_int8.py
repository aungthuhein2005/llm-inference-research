import torch

def symmetric_int8_quantization(tensor, scale):
    """
    Quantizes a tensor to symmetric int8 format.

    Args:
        tensor (torch.Tensor): The input tensor to be quantized.
        scale (float): The scale factor for quantization.

    Returns:
        torch.Tensor: The quantized tensor in int8 format.
    """
    # Ensure the input tensor is a float tensor
    if not tensor.is_floating_point():
        raise ValueError("Input tensor must be a floating point tensor.")

    # Perform symmetric quantization
    quantized_tensor = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)

    return quantized_tensor

def main():
    # Example usage
    tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32)
    scale = 0.1
    quantized_tensor = symmetric_int8_quantization(tensor, scale)
    print("Original Tensor:\n", tensor)
    print("Quantized Tensor:\n", quantized_tensor)

if __name__ == "__main__":
    main()