import numpy as np
import torch


def mean_var_normalize(explanation_tensor, eps=1e-25):
    """
    Normalize explanations using mean and variance.

    Parameters
    ----------
    explanation_tensor : Tensor
        Explanations in the form of a tensor.
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-25.

    Returns
    -------
    Tensor
        Normalized explanations.

    See Also
    --------
    median_iqr_normalize: Normalize explanations using median and interquartile range.
    second_moment_normalize: Normalize explanations using the second moment.

    Examples
    --------
    >>> import torch
    >>> explanation = torch.randn(1, 3, 64, 64)  # Input explanation tensor
    >>> normalized_explanation = mean_var_normalize(explanation)
    """
    var, mean = torch.var_mean(explanation_tensor, dim=[0, 2, 3, 4], unbiased=True, keepdim=True)
    return (explanation_tensor - mean) / torch.sqrt(var + eps)


def median_iqr_normalize(explanation_tensor, eps=1e-25):
    """
    Normalize explanations using median and interquartile range.

    Parameters
    ----------
    explanation_tensor : Tensor
        Explanations in the form of a tensor.
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-25.

    Returns
    -------
    Tensor
        Normalized explanations.

    See Also
    --------
    mean_var_normalize: Normalize explanations using mean and variance.
    second_moment_normalize: Normalize explanations using the second moment.

    Examples
    --------
    >>> import torch
    >>> explanation = torch.randn(1, 3, 64, 64)  # Input explanation tensor
    >>> normalized_explanation = median_iqr_normalize(explanation)
    """
    explanation_tensor = explanation_tensor.squeeze(0)
    median = torch.tensor(np.median(explanation_tensor, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_25 = torch.tensor(np.quantile(explanation_tensor, q=0.25, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    q_75 = torch.tensor(np.quantile(explanation_tensor, q=0.75, axis=[1, 2, 3])).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    iqr = q_75 - q_25
    return (explanation_tensor - median) / (iqr + eps)


def second_moment_normalize(explanation_tensor, eps=1e-25):
    """
    Normalize explanations using the second moment.

    Parameters
    ----------
    explanation_tensor : Tensor
        Explanations in the form of a tensor.
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-25.

    Returns
    -------
    Tensor
        Normalized explanations.

    See Also
    --------
    mean_var_normalize: Normalize explanations using mean and variance.
    median_iqr_normalize: Normalize explanations using median and interquartile range.

    Examples
    --------
    >>> import torch
    >>> explanation = torch.randn(1, 3, 64, 64)  # Input explanation tensor
    >>> normalized_explanation = second_moment_normalize(explanation)
    """
    std = torch.std(explanation_tensor, dim=[3, 4], keepdim=True)
    mean_std = torch.mean(std, dim=2, keepdim=True)
    normalized_tensor = explanation_tensor / (mean_std + eps)
    return normalized_tensor
