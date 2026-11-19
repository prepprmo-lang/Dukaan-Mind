import math


def format_indian_number(value: float, decimals: int = 0) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "—"

    formatted = f"{abs(number):.{decimals}f}"
    integer_part, decimal_separator, fractional_part = formatted.partition(".")
    if len(integer_part) > 3:
        last_group = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        integer_part = ",".join(reversed(groups)) + "," + last_group

    sign = "-" if number < 0 else ""
    fraction = f".{fractional_part}" if decimal_separator else ""
    return f"{sign}{integer_part}{fraction}"


def format_inr(value: float, decimals: int = 2) -> str:
    formatted = format_indian_number(value, decimals)
    if formatted == "—":
        return formatted
    if formatted.startswith("-"):
        return f"-₹{formatted[1:]}"
    return f"₹{formatted}"
