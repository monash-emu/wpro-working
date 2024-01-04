from typing import Dict
import numpy as np
from scipy.stats import gamma

def get_gamma_params_from_mean_sd(req_mean: float, req_sd: float) -> Dict[str, float]:
    var = req_sd ** 2.0
    scale = var / req_mean
    a = req_mean / scale
    return {'a': a, 'scale': scale}

def get_gamma_densities_from_params(mean: float, sd: float, n_times: int) -> np.array:
    """Get integrals over integer differences in gamma distribution for simulation duration.
    """
    params = get_gamma_params_from_mean_sd(mean, sd)
    return np.diff(gamma.cdf(range(n_times + 1), **params))
