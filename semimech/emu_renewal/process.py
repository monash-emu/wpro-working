from typing import List
import numpy as np
import pandas as pd


def get_interp_vals_over_model_time(
    req: List[float], 
    n_times: int,
) -> np.array:
    """Linear interpolation at requested values at regular intervals over simulation period.

    Args:
        req: Requested values
        n_times: Length of simulation, which the request will be evenly spread over

    Returns:
        Values for each of the n time values of the simulation period
    """
    return np.interp(range(n_times), np.linspace(0.0, n_times, len(req)), req)

def get_step_values_over_model_time(
    req: List[float], 
    n_times: int
) -> list:
    """Another option, but this one seems to work extremely poorly in practice.

    Args:
        req: Requested values
        n_times: Length of simulation, which the request will be evenly spread over

    Returns:
        Values for each of the n time values of the simulation period
    """
    breaks_series = pd.Series(np.linspace(0.0, n_times, len(req) + 1))
    process_vals = []
    for time in range(n_times):
        process_req_index = breaks_series[breaks_series <= time].index[-1]
        process_vals.append(req[process_req_index])
    return process_vals
