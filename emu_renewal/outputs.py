from typing import List
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from estival.sampling.tools import SampleIterator

from .renew import RenewalModel

PANEL_SUBTITLES = ["cases", "susceptibles", "R", "transmission potential"]
MARGINS = {m: 20 for m in ["t", "b", "l", "r"]}


def get_spaghetti_from_params(
    model: RenewalModel, 
    params: SampleIterator, 
    model_func: callable,
) -> pd.DataFrame:
    """Run parameters through the model to get outputs.

    Args:
        model: The renewal model
        params: The parameter sets to feed through the model
        model_func: The model function to run the parameters through

    Returns:
        Dataframe with index of model times and multiindexed columns,
            with first level being the output name and second the parameter set
            by chain and iteration
    """
    index_names = model.epoch.index_to_dti(model.model_times)
    column_names = pd.MultiIndex.from_product([params.index.map(str), PANEL_SUBTITLES])
    spaghetti = pd.DataFrame(index=index_names, columns=column_names)
    for i, p in params.iterrows():
        res = model_func(**{k: v for k, v in p.items() if "dispersion" not in k})
        spaghetti.loc[:, str(i)] = np.array([res.incidence * p["cdr"], res.suscept, res.r_t, res.process]).T
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


def get_quant_df_from_spaghetti(
    model: RenewalModel, 
    spaghetti: pd.DataFrame, 
    quantiles: List[float],
) -> pd.DataFrame:
    """Calculate requested quantiles over spaghetti created
    in previous function.

    Args:
        model: The renewal model
        spaghetti: The output of get_spaghetti_from_params
        quantiles: The quantiles at which to make the calculations

    Returns:
        Dataframe with index of model times and multiindexed columns,
            with first level being the output name and second the quantile
    """
    index_names = model.epoch.index_to_dti(model.model_times)
    column_names = pd.MultiIndex.from_product([PANEL_SUBTITLES, quantiles])
    quantiles_df = pd.DataFrame(index=index_names, columns=column_names)
    for col in PANEL_SUBTITLES:
        quantiles_df[col] = spaghetti[col].quantile(quantiles, axis=1).T
    return quantiles_df


def get_standard_four_subplots() -> go.Figure:
    """Get a figure object with standard formatting of 2 x 2 subplots.

    Returns:
        The figure object
    """
    return make_subplots(
        rows=2, 
        cols=2, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        horizontal_spacing=0.05, 
        subplot_titles=PANEL_SUBTITLES,
    )


def plot_spaghetti(
    spaghetti: pd.DataFrame, 
    targets: pd.Series,
) -> go.Figure:
    """Plot the outputs of the function that gets 
    spaghetti outputs from parameters above.

    Args:
        spaghetti: The output of get_spaghetti_from_params
        targets: The target values of the calibration algorithm

    Returns:
        The figure object
    """
    fig = get_standard_four_subplots()
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    for i in range(4):
        fig.add_traces(spaghetti[PANEL_SUBTITLES[i]].plot().data, rows=i // 2 + 1, cols=i % 2 + 1)
    return fig.update_layout(margin=MARGINS, height=600).update_yaxes(rangemode="tozero")


def get_area_from_df(
    df: pd.DataFrame,
    columns: List[float], 
    colour: str,
) -> go.Scatter:
    """Get a patch object to add to a plotly graph from a dataframe
    that contains data for the upper and lower margins.

    Args:
        df: The data
        columns: The names of the columns containing the upper and lower margins
        colour: The colour request

    Returns:
        The patch
    """
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    y_vals = df[columns[0]].to_list() + df[columns[1]][::-1].to_list()
    return go.Scatter(x=x_vals, y=y_vals, line={"width": 0.0, "color": colour}, fill="toself")


def add_ci_patch_to_plot(
    fig: go.Figure, 
    df: pd.DataFrame, 
    colour: str, 
    row: int, 
    col: int,
):
    """Add a median line and confidence interval patch to a plotly figure object.

    Args:
        fig: The figure object
        df: The data to plot
        colour: The colour request
        row: The row of the subplot figure
        col: The column of the subplot figure
    """
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    fig.add_trace(get_area_from_df(df, columns=[0.05, 0.95], colour=colour), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_vals, y=df[0.5], line={"color": colour}), row=row, col=col)


def plot_uncertainty_patches(
    quantiles: List[float], 
    targets: pd.Series, 
    colours: List[str],
) -> go.Figure:
    """Create the main uncertainty output figure for a renewal analysis.

    Args:
        quantiles: Requested quantiles
        targets: The target values of the calibration algorithm
        colour: The colour requests

    Returns:
        The figure object        
    """
    fig = get_standard_four_subplots()
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    for i in range(4):
        add_ci_patch_to_plot(fig, quantiles[PANEL_SUBTITLES[i]], colours[i], i // 2 + 1, i % 2 + 1)
    return fig.update_layout(margin=MARGINS, height=600, showlegend=False).update_yaxes(rangemode="tozero")


def plot_3d_spaghetti(
    spaghetti: pd.DataFrame, 
    column_req: List[str],
) -> go.Figure:
    """Plot to variables on y and z axes against index
    of a standard spaghetti dataframe.

    Args:
        spaghetti: Output of get_spaghetti_from_params
        column_req: The columns to plot against one-another

    Returns:
        The figure object
    """
    fig = go.Figure()
    col_1, col_2 = column_req
    for i in spaghetti.columns.get_level_values(1):
        x_data = spaghetti.index
        y_data = spaghetti[(col_1, i)]
        z_data = spaghetti[(col_2, i)]
        trace = go.Scatter3d(x=x_data, y=y_data, z=z_data, mode="lines", line={"width": 5.0})
        fig.add_trace(trace)
    axis_titles = {"xaxis": {"title": "time"}, "yaxis": {"title": col_1}, "zaxis": {"title": col_2}}
    return fig.update_layout(showlegend=False, scene=axis_titles, height=800)
