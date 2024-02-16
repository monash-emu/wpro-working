import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

PANEL_SUBTITLES = ["cases", "susceptibles", "R", "transmission potential"]
MARGINS = {m: 20 for m in ["t", "b", "l", "r"]}


def get_spaghetti_from_params(model, params, model_func):
    index_names = model.epoch.index_to_dti(model.model_times)
    column_names = pd.MultiIndex.from_product([params.index.map(str), PANEL_SUBTITLES])
    spaghetti = pd.DataFrame(index=index_names, columns=column_names)
    for i, p in params.iterrows():
        res = model_func(**{k: v for k, v in p.items() if "dispersion" not in k})
        spaghetti.loc[:, str(i)] = np.array([res.incidence * p["cdr"], res.suscept, res.r_t, res.process]).T
    spaghetti.columns = spaghetti.columns.swaplevel()
    return spaghetti.sort_index(axis=1, level=0)


def get_quant_df_from_spaghetti(model, spaghetti, quantiles):
    index_names = model.epoch.index_to_dti(model.model_times)
    column_names = pd.MultiIndex.from_product([PANEL_SUBTITLES, quantiles])
    quantiles_df = pd.DataFrame(index=index_names, columns=column_names)
    for col in PANEL_SUBTITLES:
        quantiles_df[col] = spaghetti[col].quantile(quantiles, axis=1).T
    return quantiles_df


def get_standard_four_subplots():
    return make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=PANEL_SUBTITLES)


def plot_spaghetti(spaghetti, targets):
    fig = get_standard_four_subplots()
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    for i in range(4):
        fig.add_traces(spaghetti[PANEL_SUBTITLES[i]].plot().data, rows=i // 2 + 1, cols=i % 2 + 1)
    return fig.update_layout(margin=MARGINS, height=600).update_yaxes(rangemode="tozero")


def get_area_from_df(df, columns, colour):
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    y_vals = df[columns[0]].to_list() + df[columns[1]][::-1].to_list()
    return go.Scatter(x=x_vals, y=y_vals, line={"width": 0.0, "color": colour}, fill="toself")


def add_ci_patch_to_plot(fig, df, colour, row, col):
    x_vals = df.index.to_list() + df.index[::-1].to_list()
    fig.add_trace(get_area_from_df(df, columns=[0.05, 0.95], colour=colour), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_vals, y=df[0.5], line={"color": colour}), row=row, col=col)


def plot_uncertainty_patches(quantiles, targets, colours):
    fig = get_standard_four_subplots()
    fig.add_trace(go.Scatter(x=targets.index, y=targets, mode="markers"), row=1, col=1)
    for i in range(4):
        add_ci_patch_to_plot(fig, quantiles[PANEL_SUBTITLES[i]], colours[i], i // 2 + 1, i % 2 + 1)
    return fig.update_layout(margin=MARGINS, height=600, showlegend=False).update_yaxes(rangemode="tozero")
