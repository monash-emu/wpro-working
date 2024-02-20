from summer2.functions import interpolate as sinterp
from jax import lax, numpy as jnp


def _get_cos_curve_at_x(x, xdata, ydata):
    idx = sinterp.binary_search_sum_ge(x, xdata.points) - 1

    offset = x - xdata.points[idx]
    relx = offset / xdata.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return ydata.points[idx] + (rely * ydata.ranges[idx])


def cosine_multicurve(
    t: float, xdata: sinterp.InterpolatorScaleData, ydata: sinterp.InterpolatorScaleData
):
    # Branch on whether t is in bounds
    bounds_state = sum(t > xdata.bounds)
    branches = [
        lambda _, __, ___: ydata.bounds[0],
        _get_cos_curve_at_x,
        lambda _, __, ___: ydata.bounds[1],
    ]
    return lax.switch(bounds_state, branches, t, xdata, ydata)
