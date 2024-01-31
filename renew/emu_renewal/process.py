from typing import Tuple
import numpy as np
from scipy.interpolate import CubicSpline


class InterpFunc():
    """Abstract class for interpolation functions.
    Primarily for use in creating non-mechanistic processes here.
    """
    def __init__(self, x_vals, y_vals):
        self.x_vals = x_vals
        self.y_vals = y_vals

    def get_interp_func(self):
        pass

    def get_description(self):
        pass


class LinearInterpFunc(InterpFunc):
    """Simple linear interpolation.
    """
    def get_interp_func(self):
        return lambda x: np.interp(x, self.x_vals, self.y_vals)


class SplineInterpFunc(InterpFunc):
    """Another standard interpolation option,
    which has advantages of being continuous with continuous gradient.
    """
    def get_interp_func(self):
        return CubicSpline(self.x_vals, self.y_vals)


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


class CosInterpFunc(InterpFunc):
    def get_interp_func(self):
        coords = list(zip(self.x_vals, self.y_vals))

        def piecewise_cosine_func(x):
            start_cond = x < self.x_vals[0]
            mid_conds = [self.x_vals[i] <= x < self.x_vals[i + 1] for i in range(len(coords) - 1)]
            end_cond = x >= self.x_vals[-1]
            conds = [start_cond] + mid_conds + [end_cond]

            start_func = lambda x: self.y_vals[0]
            mid_funcs = [get_cos_points_link(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
            end_func = lambda x: self.y_vals[-1]
            funcs = [start_func] + mid_funcs + [end_func]

            return np.piecewise(x, conds, funcs)

        return np.vectorize(piecewise_cosine_func)

    def get_description():
        return 'The interpolation function consisted of a piecewise function ' \
            'constructed from a cosine function on domain zero to $\pi$. ' \
            'This function was then translated and scaled vertically and horizontally ' \
            'such that the start and end points of the cosine function ' \
            '(at which the gradient is zero) pass through the two consecutive ' \
            'process values. This process was repeated for each interval for interpolation. ' \
            'This interpolation approach provides a function of time for which ' \
            'the gradient and all higher order gradients are continuous. '
