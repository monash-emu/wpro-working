from typing import List
import numpy as np

def get_interp_vals_over_model_time(
    req: List[float], 
    n_times: int,
) -> np.array:
    """Linear interpolation at requested values at regular intervals over simulation period.

    Args:
        req: Requested values
        n_times: Length of simulation, which the request will be evenly spread over

    Returns:
        Array with values for each of the n time values of the simulation period
    """
    return np.interp(range(n_times), np.linspace(0.0, n_times, len(req)), req)