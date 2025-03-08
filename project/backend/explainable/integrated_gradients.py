import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import autograd

from autoencoder import encoder, decoder

# Update this according to your preprocessed data dimensions.
INPUT_DIM = 384  # Example: update to match your input feature size

# Instantiate the encoder and decoder models.
encoder_model = encoder(INPUT_DIM)
decoder_model = decoder(INPUT_DIM)

# Optionally, load your pretrained weights here:
# encoder_model.load_state_dict(torch.load("path/to/encoder_weights.pth"))
# decoder_model.load_state_dict(torch.load("path/to/decoder_weights.pth"))

encoder_model.eval()
decoder_model.eval()


def autoencoder_model(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Passes the input through the autoencoder (encoder followed by decoder) and returns the output.
    """
    latent = encoder_model(input_tensor)
    output = decoder_model(latent)
    return output


def integrated_gradients(model, input_sample, baseline=None, steps=50):
    """
    Computes integrated gradients of the model output with respect to the input sample.
    
    Args:
        model: A callable that returns output given an input tensor.
        input_sample (torch.Tensor): The sample for which attributions are computed.
        baseline (torch.Tensor or None): Initial tensor for interpolation (default is a zero tensor).
        steps (int): Number of interpolation steps.
    
    Returns:
        torch.Tensor: Integrated gradients with the same shape as input_sample.
    """


    array = input_sample.values
    float_array = array.astype(np.float64)
    torch_dataset = torch.from_numpy(float_array).float()
    sample_transformed = autograd.Variable(torch_dataset)
    if baseline is None:
        baseline = torch.zeros_like(sample_transformed)
    # Create interpolation steps from baseline to the input sample.
    scaled_inputs = [
        baseline + (float(i) / steps) * (sample_transformed - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_(True)
        output = model(scaled_input)
        loss = loss_function(output, scaled_input)
        loss.backward()
        grads.append(scaled_input.grad.detach())

    # Average the gradients and scale with the input difference.
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grads = (sample_transformed - baseline) * avg_grads
    return integrated_grads


def get_top_attributions(attributions: torch.Tensor, n: int):
    """
    Extracts the top n features based on the absolute integrated gradient values.

    Args:
        attributions (torch.Tensor): The computed integrated gradients of shape [1, num_features].
        n (int): Number of top features to extract.

    Returns:
        List of tuples: Each tuple contains (feature_index, attribution_value).
    """
    # Get the absolute attribution values and remove extra dimensions.
    attributions_abs = attributions.abs().detach().cpu().squeeze().numpy()
    # Get indices for the top n attributions (largest absolute values)
    top_indices = np.argsort(attributions_abs)[-n:][::-1]
    top_values = attributions_abs[top_indices]
    # Create a list of tuples (feature_index, value)
    return pd.Series(top_values, index=top_indices)


