import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_output_fit(
    targets: pd.Series,
    result,
    n_times: int,
    cdr: float = 1.0,
) -> go.Figure:
    """Plot results from a fitting attempt against the target data and show estimated random process.

    Args:
        targets: Data targeted
        result: Epidemiological outputs by the renewal process
        process_vals: Estimated non-mechanistic variation in infectiousness
        n_times: Number of simulated time points

    Returns:
        Interactive figure
    """
    fitted, suscept, r_t, process_vals = result
    model_times = pd.Series(range(n_times))
    fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=['incidence', 'reproduction number', 'susceptibles'])
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode='markers', name='targets'), row=1, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=fitted * cdr, name='model'), row=1, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=process_vals, name='transmission potential'), row=2, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=r_t, name='Rt'), row=2, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=suscept, name='susceptibles'), row=3, col=1)
    return fig.update_layout(margin={'t': 20, 'b': 5, 'l': 5, 'r': 5})
