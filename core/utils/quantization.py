from typing import List, Optional, Union
import numpy as np
import torch

from core.model import Model

# according to stephen, has code to perform hadamard transform, will use GPT-generated in the interim
def hadamard(n):
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

    H = torch.ones((1, 1), dtype=torch.float32)

    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)

    return H / n**0.5


def quantize_parameters(model: Model, method: str = "hadamard", bits: int = 8, block_size: Optional[int] = None, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    assert 2 <= bits <= 8

    # want the hypernetwork parameters specifically
    parameters = torch.nn.utils.parameters_to_vector(model.network.hypernet.parameters())
    parameters = parameters.detach().to(device="cpu", dtype=torch.float32).clone()

    # pad to a multiple of two for Hadamard transform
    parameters_size = parameters.numel()
    if method == "hadamard":
        padded_size = 2 ** int(np.ceil(np.log2(parameters_size)))
    else:
        padded_size = parameters_size

    padding = torch.zeros(padded_size - parameters_size, dtype=torch.float32)
    parameters = torch.cat([parameters, padding], dim=0)

    #construct the randomized Hadamard transform matrix
    if method == "hadamard":
        H = hadamard(padded_size)
        generator = torch.Generator().manual_seed(seed)
        D = torch.randint(0, 2, (padded_size,), generator=generator, dtype=torch.float32) * 2 - 1
        D_params = D * parameters
        parameters = torch.matmul(H, D_params)

    # quantization step
    scale = torch.max(torch.abs(parameters))
    if scale != 0:
        parameters = torch.round((parameters / scale) * (2 ** (bits - 1) - 1))
    parameters = parameters.to(torch.int8)
    return parameters, scale, parameters_size

def dequantize_parameters(quantized_parameters: torch.Tensor, scale: torch.Tensor, original_size: int, method: str = "hadamard",
                          bits: int = 8, seed: int = 0) -> torch.Tensor:
    assert 2 <= bits <= 8
    # dequantization step
    dequantized_parameters = quantized_parameters.to(torch.float32)
    dequantized_parameters = dequantized_parameters * scale / (2 ** (bits - 1) - 1)

    # regenerate the random signs for the Hadamard transform
    if method == "hadamard":
        padded_size = dequantized_parameters.numel()
        H = hadamard(padded_size)
        generator = torch.Generator().manual_seed(seed)  # Use the same seed as during quantization
        D = torch.randint(0, 2, (padded_size,), generator=generator, dtype=torch.float32) * 2 - 1
        H_params = torch.matmul(H, dequantized_parameters)
        dequantized_parameters = D * H_params  # Apply the random signs

    # reshape to the original size
    dequantized_parameters = dequantized_parameters[:original_size]

    return dequantized_parameters

