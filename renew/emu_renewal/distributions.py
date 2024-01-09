from typing import Dict
import numpy as np
from scipy.stats import gamma


def get_gamma_params_from_mean_sd(
    req_mean: float,
    req_sd: float,
) -> Dict[str, float]:
    """Calculate the parameters to construct a gamma distribution
    given user requests specified as mean and SD.

    Args:
        req_mean: Requested mean
        req_sd: Rqeuested standard deviation

    Returns:
        The two parameters (a and scale) as a dictionary
    """
    var = req_sd ** 2.0
    scale = var / req_mean
    a = req_mean / scale
    return {'a': a, 'scale': scale}

def get_gamma_densities_from_params(
    mean: float,
    sd: float,
    n_times: int,
) -> np.array:
    """Get integrals over integer differences in gamma distribution
    for simulation duration.

    Args:
        req_mean: Requested mean
        req_sd: Rqeuested standard deviation
        n_times: Number of times needed

    Returns:
        Array of differences in the CDF function
    """
    params = get_gamma_params_from_mean_sd(mean, sd)
    return np.diff(gamma.cdf(range(n_times + 1), **params))
