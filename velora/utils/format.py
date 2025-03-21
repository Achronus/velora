def number_to_short(value: int) -> str:
    """
    Converts a number into a human-readable format like `1M` or `1.25K`.

    Parameters:
        value (int): The number to convert

    Returns:
        str: The shortened version as a string
    """
    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]

    for threshold, suffix in suffixes:
        if value >= threshold:
            short_value = round(value / threshold, 2)
            return f"{short_value:g}{suffix}"

    return str(value)
