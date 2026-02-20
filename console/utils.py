from typing import List

from colour import Color


def apply_style(text: str, style: str):
    return f"[{style}]{text}[/{style}]"

def header(text):
    return apply_style(text, "rule.text")

def success(text):
    return apply_style(text, "success")

def error(text):
    return apply_style(text, "error.content")

def warning(text):
    return apply_style(text, "warning.content")

def info(text):
    return apply_style(text, "info.content")

def emphasis(text):
    return apply_style(text, "magenta")

def subtle(text):
    return apply_style(text, "med_grey")

def calc_color_gradient(color_begin: str, color_end: str, num_items: int) -> List[str]:
    if num_items < 2:
        return [color_begin]

    start_color = Color(color_begin)
    end_color = Color(color_end)
    gradient_objects = list(start_color.range_to(end_color, num_items))
    gradient_hex = [color.hex_l.upper() for color in gradient_objects]
    return gradient_hex
