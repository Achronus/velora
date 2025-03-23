def number_to_short(value: int) -> str:
    """
    Converts a number into a human-readable format like `1M` or `1.25K`.

    Parameters:
        value (int): The number to convert

    Returns:
        str: The shortened version as a string
    """
    is_negative = value < 0
    abs_value = abs(value)

    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]

    for threshold, suffix in suffixes:
        if abs_value >= threshold:
            short_value = round(abs_value / threshold, 2)
            result = f"{short_value:g}{suffix}"
            return f"-{result}" if is_negative else result

    return str(value)
