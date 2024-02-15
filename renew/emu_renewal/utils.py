def format_date_for_str(date):
    ord_excepts = {1: "st", 2: "nd", 3: "rd"}
    ordinal = ord_excepts.get(date.day % 10, "th")
    return date.strftime(f"%-d{ordinal} %B %Y")
