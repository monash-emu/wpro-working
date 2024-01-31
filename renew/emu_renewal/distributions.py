import numpy as np
from scipy.stats import gamma


class Dens():
    def __init__(self):
        pass

    def get_params():
        pass

    def get_densitites():
        pass


class GammaDens():
    def get_params(self, mean, sd):
        var = sd ** 2.0
        scale = var / mean
        a = mean / scale
        return {'a': a, 'scale': scale}
    
    def get_densities(self, n_times, req_mean, req_sd):
        return np.diff(gamma.cdf(range(n_times + 1), **self.get_params(req_mean, req_sd)))

    def describe_dens(self):
        return '\n\n### Generation times\n' \
            'Generation times for each day are calculated by ' \
            'first finding the parameters needed to construct ' \
            'a gamma distribution with mean and standard deviation ' \
            'equal to those specified by the submitted parameter values. ' \
            'The integrals of the probability density of this distribution ' \
            'between consecutive integer values are then calculated for ' \
            'later combination with the incidence time series. '
    