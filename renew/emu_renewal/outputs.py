from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_spaghetti(cases, targets, proc, suscept, r, margins, titles):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=titles)
    fig.add_traces(cases.plot().data, rows=1, cols=1)
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    fig.add_traces(proc.plot().data, rows=2, cols=1)
    fig.add_traces(suscept.plot().data, rows=1, cols=2)
    fig.add_traces(r.plot().data, rows=2, cols=2)
    return fig.update_layout(margin=margins, height=600)


def get_plotly_area_from_df(df, columns):
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    y_vals = df[columns[0]].to_list() + df[columns[1]][::-1].to_list()
    return go.Scatter(x=x_vals, y=y_vals, fill="toself")


def plot_uncertainty_patches(cases, targets, proc, suscept, r, margins, titles):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=titles)
    x_vals = cases.index
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    fig.add_trace(get_plotly_area_from_df(cases, columns=[0.05, 0.95]), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=cases[0.5]), row=1, col=1)
    fig.add_trace(get_plotly_area_from_df(suscept, columns=[0.05, 0.95]), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=suscept[0.5]), row=1, col=2)
    fig.add_trace(get_plotly_area_from_df(r, columns=[0.05, 0.95]), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=r[0.5]), row=2, col=1)
    fig.add_trace(get_plotly_area_from_df(proc, columns=[0.05, 0.95]), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=proc[0.5]), row=2, col=2)
    return fig.update_layout(margin=margins, height=600, showlegend=False)
