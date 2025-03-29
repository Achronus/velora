from math import ceil


def closest_divisible(target: int, divisor: int) -> int:
    """
    Calculates the closest divisible value between a target and divisor.

    Parameter:
        target (int): number to find divisor upper bound
        divisor (int): number to divide target by

    Returns:
        value (int): closest divisible value
    """
    quotient = ceil(target / divisor)
    return quotient * divisor


def get_mini_batch_size(buffer_size: int, batch_size: int) -> int:
    """
    Computes the mini-batch size between a buffer and batch size.

    Parameters:
        buffer_size (int): total number of steps in the buffer
        batch_size (int): size of each mini-batch for training

    Returns:
        size (int): number of mini-batches

    Raises:
        ValueError: when size mismatch, non-positive numbers, or batch size is larger than buffer size.
    """
    if buffer_size <= 0 or batch_size <= 0:
        raise ValueError("'buffer_size' and 'batch_size' must be positive numbers.")

    if batch_size > buffer_size:
        raise ValueError(f"'{batch_size=}' cannot be larger than '{buffer_size=}.")

    power_of_2 = [2**i for i in range(5, 11)]  # [32, 64, 128, 256, 512, 1024]
    num_mini_batches = buffer_size // batch_size
    remainder = buffer_size % batch_size

    if remainder == 0:
        return num_mini_batches

    suggestions = [
        bs for bs in power_of_2 if buffer_size % bs == 0 and bs <= buffer_size
    ]

    if not suggestions:
        raise ValueError(
            f"'{batch_size=}' is not evenly divisible by '{buffer_size=}'.\n"
            f"Recommended batch sizes: '{suggestions}'."
        )

    if len(suggestions) == 0:
        raise ValueError("Try a larger 'buffer_size' that is divisible by 2.")

    raise ValueError(
        f"Incompatible '{batch_size=}' with '{buffer_size=}'.\nRecommended batch sizes: '{suggestions}'."
    )
