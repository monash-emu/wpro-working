import numpy as np
from datetime import datetime


def format_date_for_str(
    date: datetime,
    include_year: bool=True,
) -> str:
    """Get a markdown-ready string that could be included in
    paragraph text from a datetime object.

    Args:
        date: The datetime object

    Returns:
        The formatted string
    """
    ord_excepts = {1: "st", 2: "nd", 3: "rd"}
    ordinal = ord_excepts.get(date.day % 10, "th")
    if include_year:
        return f"{date.day}<sup>{ordinal}</sup> {date: %B} {date: %Y}"
    else:
        return f"{date.day}<sup>{ordinal}</sup> {date: %B}"


def round_sigfig(
    value: float, 
    sig_figs: int
) -> float:
    """
    Round a number to a certain number of significant figures, 
    rather than decimal places.
    
    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    return round(value, -int(np.floor(np.log10(abs(value)))) + sig_figs - 1) if value != 0.0 else 0.0


def get_proc_period_from_index(
    idx: int, 
    model,
) -> str:
    """Get markdown-formatted string for date of
    variable process period from its index number.

    Args:
        idx: The sequence of the process period
        model: The renewal model

    Returns:
        The formatted string
    """
    start = int(model.x_proc_vals[idx])
    end = start + model.proc_update_freq - 1
    start_date = format_date_for_str(model.epoch.number_to_datetime(start), include_year=False)
    end_date = format_date_for_str(model.epoch.number_to_datetime(end), include_year=False)
    return f"Variable process update, {start_date} to {end_date}"


map_dict = {
    "cdr": "Case detection proportion",
    "gen_mean": "Generation time, mean",
    "gen_sd": "Generation time, standard deviation",
    "proc_dispersion": "Variable process update dispersion",
    "dispersion": "Data comparison dispersion",
    "rt_init": "Rt starting value",
}


def get_adjust_idata_index(
    model,
) -> callable:
    """Get function to adjust the dataframe index
    containing the model parameters.

    Args:
        model: The model

    Returns:
        The adjuster function
    """
    def adjust_idata_index(i):
        if i.startswith("proc["):
            i_proc = int(i[i.find("[") + 1: i.find("]")])
            return get_proc_period_from_index(i_proc, model)
        elif i in map_dict:
            return map_dict[i]
        else:
            raise ValueError("Parameter not found")
    return adjust_idata_index


col_names_map = {
    "sd": "standard deviation",
    "hdi_3%": "high-density interval, 3%",
    "hdi_97%": "high-density interval, 97%",
    "ess_bulk": "effective sample size, bulk",
    "ess_tail": "effective sample size, tail",
    "r_hat": "_&#x0052;&#x0302;_",
}


def adjust_summary_cols(summary):
    summary = summary.rename(columns=col_names_map)
    summary = summary.drop(["mcse_mean", "mcse_sd"], axis=1)
    summary.columns = summary.columns.str.capitalize()
    return summary
