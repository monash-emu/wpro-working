import numpy as np
from datetime import datetime


def format_date_for_str(
    date: datetime,
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
    return f"{date.day}<sup>{ordinal}</sup> {date: %B} {date: %Y}"


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
