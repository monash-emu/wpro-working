from typing import Tuple
import numpy as np
from scipy.interpolate import CubicSpline


def get_linear_interp_func(
    req_x_vals: np.array,
    req_y_vals: np.array, 
) -> callable:
    """Linear interpolation at requested values at regular intervals over simulation period.

    Args:
        req_x_vals: Requested x-values
        req_y_vals: Requested y-values

    Returns:
        The function
    """
    return lambda x: np.interp(x, req_x_vals, req_y_vals)

def get_spline_interp_func(
    req_x_vals: np.array,
    req_y_vals: np.array, 
) -> callable:
    """Another standard interpolation option,
    which has advantages of being continuous with continuous gradient.

    Args:
        req_x_vals: Requested x-values
        req_y_vals: Requested y-values

    Returns:
        The function
    """
    return CubicSpline(req_x_vals, req_y_vals)

def get_cos_points_link(
    coords_a: Tuple[float], 
    coords_b: Tuple[float],
) -> callable:
    if coords_a[0] == coords_b[0]:
        raise ValueError('Same x-value for both submitted coordinates')
    period_adj = np.pi / (coords_b[0] - coords_a[0])
    amplitude = coords_b[1] - coords_a[1]
    def cos_link(x):
        return (-np.cos((x - coords_a[0]) * period_adj) + 1.0) / 2.0 * amplitude + coords_a[1]
    return cos_link

def get_piecewise_cosine(x_vals, y_vals):
    coords = list(zip(x_vals, y_vals))
    def piecewise_cosine_func(x):
        start_cond = x < x_vals[0]
        mid_conds = [x_vals[i] <= x < x_vals[i + 1] for i in range(len(coords) - 1)]
        end_cond = x >= x_vals[-1]
        conds = [start_cond] + mid_conds + [end_cond]

        start_func = lambda x: y_vals[0]
        mid_funcs = [get_cos_points_link(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
        end_func = lambda x: y_vals[-1]
        funcs = [start_func] + mid_funcs + [end_func]
        
        return np.piecewise(x, conds, funcs)
    return np.vectorize(piecewise_cosine_func)