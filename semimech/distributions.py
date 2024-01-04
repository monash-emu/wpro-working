from typing import Dict

def get_gamma_params_from_mean_sd(req_mean: float, req_sd: float) -> Dict[str, float]:
    var = req_sd ** 2.0
    scale = var / req_mean
    a = req_mean / scale
    return {'a': a, 'scale': scale}