from typing import Dict
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from collections import namedtuple


Outputs = namedtuple('outputs', ['incidence', 'suscept', 'r_t'])


def plot_output_fit(targets: Dict[int, float], result, process_vals, n_times: int):
    """Plot results from a fitting attempt against the target data and show estimated random process.
    """
    fitted, suscept, r_t = result
    model_times = pd.Series(range(n_times))
    fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=['incidence', 'reproduction number', 'susceptibles'])
    fig.add_trace(go.Scatter(x=list(targets.keys()), y=list(targets.values()), mode='markers', name='targets'), row=1, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=fitted, name='model'), row=1, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=process_vals, name='transmission potential'), row=2, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=r_t, name='Rt'), row=2, col=1)
    fig.add_trace(go.Scatter(x=model_times, y=suscept, name='susceptibles'), row=3, col=1)
    return fig.update_layout(height=800, margin={'t': 20, 'b': 5, 'l': 5, 'r': 5})
