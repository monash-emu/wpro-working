from typing import List
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def get_interp_vals_over_model_time(
    req_x_vals: List[float],
    req_y_vals: List[float], 
) -> callable:
    """Linear interpolation at requested values at regular intervals over simulation period.

    # Args:
    #     req: Requested values
    #     n_times: Length of simulation, which the request will be evenly spread over

    # Returns:
    #     Values for each of the n time values of the simulation period
    """
    return lambda x: np.interp(x, req_x_vals, req_y_vals)

def get_spline_values_over_model_time(
    req_x_vals: List[float],
    req_y_vals: List[float], 
) -> callable:
    """Another standard interpolation option, works better.

    # Args:
    #     req: Requested values
    #     n_times: Length of simulation, which the request will be evenly spread over

    # Returns:
    #     Values for each of the n time values of the simulation period
    """
    return CubicSpline(req_x_vals, req_y_vals)
