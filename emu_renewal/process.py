from jax import lax, numpy as jnp

from summer2.functions.interpolate import InterpolatorScaleData
from summer2.functions import interpolate as sinterp


def _get_cos_curve_at_x(
    x: float,
    xdata: InterpolatorScaleData,
    ydata: InterpolatorScaleData,
) -> float:
    """Get interpolated function value using half-cosine function.

    Args:
        x: Independent value to calculate result at
        xdata: Requested series of independent values
        ydata: Requested series of dependent values

    Returns:
        Interpolated value
    """
    idx = sinterp.binary_search_sum_ge(x, xdata.points) - 1
    offset = x - xdata.points[idx]
    relx = offset / xdata.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return ydata.points[idx] + (rely * ydata.ranges[idx])


def _get_linear_curve_at_x(
    x: float,
    xdata: InterpolatorScaleData,
    ydata: InterpolatorScaleData,
) -> float:
    """Get interpolated function value using half-cosine function.

    Args:
        x: Independent value to calculate result at
        xdata: Requested series of independent values
        ydata: Requested series of dependent values

    Returns:
        Interpolated value
    """
    idx = sinterp.binary_search_sum_ge(x, xdata.points) - 1
    offset = x - xdata.points[idx]
    relx = offset / xdata.ranges[idx]
    return ydata.points[idx] + (relx * ydata.ranges[idx])


class MultiCurve:
    """Abstract class for fitting a curve to a series of data."""

    def get_multicurve(self):
        pass

    def get_description(self):
        pass


class CosineMultiCurve(MultiCurve):
    """Fit a cosine-based curve to a series of data.
    See get_description below for details.

    Args:
        MultiCurve: Abstract parent class
    """

    def get_multicurve(
        self,
        t: float,
        xdata: InterpolatorScaleData,
        ydata: InterpolatorScaleData,
    ):
        """Construct a half-cosine-based multi-curve.

        Args:
            t: Model time
            xdata: Values of independent variable
            ydata: Values of dependent variable

        Returns:
            Curve fitting function
        """
        # Branch on whether t is in bounds
        bounds_state = sum(t > xdata.bounds)
        branches = [
            lambda _, __, ___: ydata.bounds[0],
            _get_cos_curve_at_x,
            lambda _, __, ___: ydata.bounds[1],
        ]
        return lax.switch(bounds_state, branches, t, xdata, ydata)

    def get_description(self):
        return (
            "Fitting is implemented using a half-cosine interpolation function "
            "that is translated and scaled to reach each of the points specified. "
            "This results in a function that joins each two successive requested values "
            "with a function that scales smoothly from "
            "a gradient of zero at the preceding point to "
            "a gradient of zero at the subsequent point. "
        )


class LinearMultiCurve(MultiCurve):
    """Fit a cosine-based curve to a series of data.
    See get_description below for details.

    Args:
        MultiCurve: Abstract parent class
    """

    def get_multicurve(
        self,
        t: float,
        xdata: InterpolatorScaleData,
        ydata: InterpolatorScaleData,
    ):
        """Construct a half-cosine-based multi-curve.

        Args:
            t: Model time
            xdata: Values of independent variable
            ydata: Values of dependent variable

        Returns:
            Curve fitting function
        """
        # Branch on whether t is in bounds
        bounds_state = sum(t > xdata.bounds)
        branches = [
            lambda _, __, ___: ydata.bounds[0],
            _get_linear_curve_at_x,
            lambda _, __, ___: ydata.bounds[1],
        ]
        return lax.switch(bounds_state, branches, t, xdata, ydata)

    def get_description(self):
        return (
            "Fitting is implemented using a half-cosine interpolation function "
            "that is translated and scaled to reach each of the points specified. "
            "This results in a function that joins each two successive requested values "
            "with a function that scales smoothly from "
            "a gradient of zero at the preceding point to "
            "a gradient of zero at the subsequent point. "
        )
