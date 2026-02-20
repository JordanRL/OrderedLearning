from dataclasses import dataclass
from enum import Enum


class TimeFormat(Enum):
    """Time display format options for console timestamps."""
    DEFAULT = "%I:%M:%S %p"              # 12-hour with AM/PM
    NO_AM_PM = "%I:%M:%S"                # 12-hour without AM/PM
    NO_SECONDS = "%I:%M %p"              # 12-hour without seconds
    NO_SECONDS_NO_AM_PM = "%I:%M"        # 12-hour minimal
    TWENTY_FOUR_HOUR = "%H:%M:%S"        # 24-hour with seconds
    TWENTY_FOUR_HOUR_NO_SECONDS = "%H:%M" # 24-hour without seconds


class ConsoleMode(Enum):
    """Console operation mode."""
    LIVE = "live"       # Full live UI with layouts
    NORMAL = "normal"   # Standard Rich console output
    LOGGING = "logging" # Output to log file only
    NULL = "null"       # No output at all
    SILENT = "silent"   # Progress bar only; all text output suppressed


class ColorSystem(Enum):
    """Color system configuration for the console."""
    AUTO = "auto"
    STANDARD = "standard"
    COLOR_256 = "256"
    TRUECOLOR = "truecolor"
    WINDOWS = "windows"


@dataclass
class ConsoleConfig:
    """
    Configuration for the OLConsole display system.

    :ivar mode: The operating mode for the console (LIVE, NORMAL, LOGGING, NULL).
    :ivar use_live_display: Whether to use Rich Live display for real-time updates.
    :ivar use_unicode: Whether to use Unicode characters in output.
    :ivar use_colors: Whether to use colored output.
    :ivar use_stats: Whether to display the stats panel in LIVE mode.
    :ivar show_time: Whether to show timestamps on messages.
    :ivar time_format: Format for timestamp display.
    :ivar timezone: Timezone for timestamp display (e.g., "UTC", "America/New_York").
    :ivar log_file: Path to log file when using LOGGING mode.
    :ivar theme_name: Name of the theme to use.
    :ivar color_system: Color system to use for output.
    """
    mode: ConsoleMode = ConsoleMode.LIVE
    use_live_display: bool = True
    use_unicode: bool = True
    use_colors: bool = True
    use_stats: bool = True
    show_time: bool = True
    time_format: TimeFormat = TimeFormat.DEFAULT
    timezone: str = "UTC"
    log_file: str | None = None
    theme_name: str = "default"
    color_system: ColorSystem = ColorSystem.TRUECOLOR
