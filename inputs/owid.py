import pandas as pd


def load_country_owid_data(
    country_iso: str, 
    indicators: str,
) -> pd.DataFrame:
    """Load single indicator for single country from main OWID spreadsheet.

    Args:
        country_iso: ISO code identifying the country
        indicators: Name of the indicator or indicators

    Returns:
        The data
    """
    data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv', index_col='date', low_memory=False)
    data.index = pd.to_datetime(data.index)
    return data[data['iso_code'] == country_iso][indicators]
