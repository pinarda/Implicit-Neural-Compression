from importlib.metadata import metadata
from typing import List, Optional, Union
import numpy as np
import torch
from torch import distributions as dist

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

def fwht(x):
    # get the shape of the input tensor
    # shape = x.shape
    m, n = x.shape
    if 2 ** int(np.log2(n)) != n:
        raise ValueError("The second dimension must be a power of 2.")

    y = x.clone().detach()
    for i in range(1, int(np.log2(n))+1):
        k = 2 ** i
        k2 = 2 ** (i - 1)

        y = torch.reshape(y, (m, -1, k))
        tmp = y[:, :,0:k2].clone()
        bot = y[:, :,k2:k].clone()
        y[:, :, 0:k2] = tmp + bot
        y[:, :, k2:k] = tmp - bot
        y = torch.reshape(y, x.shape)

    return y/n**0.5

def gaussian_compand(x: torch.Tensor, mu: torch.Tensor = 0.0, sigma: torch.Tensor = 1.0) -> torch.Tensor:
    transformed = dist.Normal(mu, sigma).cdf(x)
    return transformed

def inverse_gaussian_compand(y: torch.Tensor, mu: torch.Tensor = 0.0, sigma: torch.Tensor = 1.0) -> torch.Tensor:
    inverse_transformed = dist.Normal(mu, sigma).icdf(y)
    return inverse_transformed

def quantize_parameters(model: Model, method: str = "hadamard", bits: int = 8, block_size: Optional[int] = None, seed: int = 0, compand=True) -> tuple[torch.Tensor, dict]:
    assert 2 <= bits <= 8
    metadata = {}
    # want the hypernetwork parameters specifically
    parameters = torch.nn.utils.parameters_to_vector(model.network.hypernet.parameters())
    parameters = parameters.detach().to(device="cpu", dtype=torch.float32).clone()

    # pad to a multiple of two for Hadamard transform
    parameters_size = parameters.numel()
    metadata["parameters_size"] = parameters_size
    if method == "hadamard" or method == "fwht":
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

    if method == "fwht":
        generator = torch.Generator().manual_seed(seed)
        D = torch.randint(0, 2, (padded_size,), generator=generator, dtype=torch.float32) * 2 - 1
        D_params = D * parameters
        parameters = fwht(D_params.reshape(1, -1)).flatten()

    # if companding is desired, apply the Gaussian companding function
    if compand:
        mu = parameters.mean()
        sigma = parameters.std(unbiased=False)
        metadata["mu"] = mu.item()
        metadata["sigma"] = sigma.item()
        parameters = gaussian_compand(parameters, mu, sigma)

        parameters = torch.clamp(
            torch.floor(parameters * (2 ** bits)),
            0,
            2 ** bits - 1
        )
        parameters = parameters.to(torch.uint8)
    else:
        scale = torch.max(torch.abs(parameters))
        metadata["scale"] = scale.item()
        if scale != 0:
            parameters = torch.round((parameters / scale) * (2 ** (bits - 1) - 1))
        parameters = parameters.to(torch.int8)
    return parameters, metadata

def dequantize_parameters(quantized_parameters: torch.Tensor, metadata: dict, method: str = "hadamard",
                          bits: int = 8, seed: int = 0, compand=True) -> torch.Tensor:
    assert 2 <= bits <= 8
    # dequantization step
    dequantized_parameters = quantized_parameters.to(torch.float32)

    # if companding was applied, apply the inverse Gaussian companding function
    if compand:
        shifted_parameters = (dequantized_parameters + 0.5) / (2 ** bits)
        dequantized_parameters = inverse_gaussian_compand(shifted_parameters, metadata.get("mu", 0.0), metadata.get("sigma", 1.0))
    else:
        dequantized_parameters = dequantized_parameters * (metadata.get("scale", 1.0) / (2 ** (bits - 1) - 1))

    # regenerate the random signs for the Hadamard transform
    if method == "hadamard":
        padded_size = dequantized_parameters.numel()
        H = hadamard(padded_size)
        generator = torch.Generator().manual_seed(seed)  # Use the same seed as during quantization
        D = torch.randint(0, 2, (padded_size,), generator=generator, dtype=torch.float32) * 2 - 1
        H_params = torch.matmul(H, dequantized_parameters)
        dequantized_parameters = D * H_params  # Apply the random signs

    if method == "fwht":
        padded_size = dequantized_parameters.numel()
        generator = torch.Generator().manual_seed(seed)
        D = torch.randint(0, 2, (padded_size,), generator=generator, dtype=torch.float32) * 2 - 1
        H_params = fwht(dequantized_parameters.reshape(1, -1)).flatten()
        dequantized_parameters = D * H_params

    # reshape to the original size
    dequantized_parameters = dequantized_parameters[:metadata.get("parameters_size")]

    return dequantized_parameters

