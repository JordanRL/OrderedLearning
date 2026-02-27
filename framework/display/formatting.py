"""Value formatters for display output.

Small helper functions that convert metric values into Rich-styled
strings used by the lifecycle and results display functions.
"""


def format_prob(value, baseline=None):
    """Format a probability value: '3.2e-04 (1.6x baseline)'."""
    s = f"{value:.2e}"
    if baseline and baseline > 0:
        ratio = value / baseline
        s += f" ({ratio:.1f}x baseline)"
    return s


def format_change(init, final, higher_is_better=True):
    """Format a change: '2.1x' in green or '0.3x' in red."""
    if init == 0:
        return "[placeholder]\u2014[/placeholder]"

    if higher_is_better:
        ratio = final / init
    else:
        ratio = init / final  # For loss: lower is better

    if ratio >= 1.0:
        return f"[metric.improved]{ratio:.1f}x[/metric.improved]"
    else:
        return f"[metric.degraded]{ratio:.1f}x[/metric.degraded]"


def format_accuracy(value):
    """Format accuracy with graduated color coding."""
    if value >= 99:
        return f"[accuracy.excellent]{value:.2f}%[/accuracy.excellent]"
    elif value >= 80:
        return f"[accuracy.good]{value:.2f}%[/accuracy.good]"
    elif value >= 50:
        return f"[accuracy.fair]{value:.2f}%[/accuracy.fair]"
    else:
        return f"[accuracy.poor]{value:.2f}%[/accuracy.poor]"


def format_loss(value):
    """Format loss value."""
    return f"[metric.value]{value:.4f}[/metric.value]"
