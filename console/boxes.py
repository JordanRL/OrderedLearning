"""Custom Rich Box styles for the OLConsole display system."""

from rich.box import Box

EMPTY_BOX = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

BOTTOM_BORDER = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "━━━━\n"
)

TOP_BORDER = Box(
    "━━━━\n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

TOP_BOTTOM_BORDER = Box(
    "━━━━\n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "━━━━\n"
)

BOTTOM_PADDED_BORDER = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    " ━━ \n"
)

TOP_PADDED_BORDER = Box(
    " ━━ \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

TOP_BOTTOM_PADDED_BORDER = Box(
    " ━━ \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    " ━━ \n"
)
