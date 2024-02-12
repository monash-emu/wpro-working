from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_spaghetti(cases, targets, proc, suscept, r, margins, titles):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=titles)
    fig.update_layout(margin=margins, height=600)
    fig.add_traces(cases.plot().data, rows=1, cols=1)
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    fig.add_traces(proc.plot().data, rows=2, cols=1)
    fig.add_traces(suscept.plot().data, rows=1, cols=2)
    fig.add_traces(r.plot().data, rows=2, cols=2)
    return fig


def plot_uncertainty_patches(cases, select_data, proc, suscept, r, margins, titles):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=titles)
    fig.update_layout(margin=margins, height=600, showlegend=False)
    x_vals = proc.index
    fig.add_trace(go.Scatter(x=select_data.index, y=select_data, mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=cases[0.05], line={"width": 0.0}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=cases[0.95], fill="tonexty", mode="none"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=cases[0.5], line={"width": 1.0}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=suscept[0.05], line={"width": 0.0}), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=suscept[0.95], fill="tonexty", mode="none"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=suscept[0.5], line={"width": 1.0}), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=r[0.05], line={"width": 0.0}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=r[0.95], fill="tonexty", mode="none"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=r[0.5], line={"width": 1.0}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=proc[0.05], line={"width": 0.0}), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=proc[0.95], fill="tonexty", mode="none"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=proc[0.5], line={"width": 1.0}), row=2, col=2)
    return fig
