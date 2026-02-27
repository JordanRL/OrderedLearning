"""Display test fixtures — capture console output for assertion."""

import io

import pytest
from rich.console import Console

from console.config import ConsoleConfig, ConsoleMode
from console.olconsole import OLConsole
from console.themes import OLDarkTheme


@pytest.fixture
def capture_console():
    """Swap OLConsole to NORMAL mode with a StringIO buffer.

    Yields a callable that returns the captured output as a string.
    Restores the original NULL-mode console on teardown.
    """
    console = OLConsole()
    original_console = console._console
    original_mode = console._mode

    buffer = io.StringIO()
    console._console = Console(
        file=buffer, width=120, highlight=False, no_color=True,
        theme=OLDarkTheme(),
    )
    console._mode = ConsoleMode.NORMAL

    def get_output():
        return buffer.getvalue()

    yield get_output

    # Restore original state
    console._console = original_console
    console._mode = original_mode
