from typing import Dict
import numpy as np
from scipy.stats import gamma
from collections import namedtuple

DistParams = namedtuple('dist_params', ['parameters', 'description'])
DistDensVals = namedtuple('dist_vals', ['vals', 'description'])


def get_gamma_params_from_mean_sd(
    req_mean: float,
    req_sd: float,
) -> Dict[str, float]:
    """See desc object below.

    Args:
        req_mean: Requested mean
        req_sd: Rqeuested standard deviation

    Returns:
        The two parameters (a and scale) as a dictionary
    """
    desc = '\n\n### Generation times\n' \
        'Generation times for each day are calculated by ' \
        'first finding the parameters needed to construct ' \
        'a gamma distribution with mean and standard deviation ' \
        'equal to those specified by the submitted parameter values. '
    var = req_sd ** 2.0
    scale = var / req_mean
    a = req_mean / scale
    return DistParams({'a': a, 'scale': scale}, desc)

def get_gamma_densities_from_params(
    req_mean: float,
    req_sd: float,
    n_times: int,
) -> np.array:
    """See desc object below.

    Args:
        req_mean: Requested mean
        req_sd: Rqeuested standard deviation
        n_times: Number of times needed

    Returns:
        Array of differences in the CDF function
    """
    desc = 'The integrals of the probability density of this distribution ' \
        'between consecutive integer values are then calculated for ' \
        'later combination with the incidence time series. '
    params = get_gamma_params_from_mean_sd(req_mean, req_sd)
    return DistDensVals(np.diff(gamma.cdf(range(n_times + 1), **params.parameters)), params.description + desc)
