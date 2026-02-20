import sys
import time
from typing import Any, List, Dict, Literal
from zoneinfo import ZoneInfo
import traceback
from traceback import FrameSummary

import pyperclip
from pyfiglet import Figlet
from readchar import readkey, key
from rich.console import Console
from rich.align import Align
from rich.box import SQUARE
from rich.console import RenderableType, Group
from rich.layout import Layout
from rich.live import Live
from rich.measure import Measurement
from rich.panel import Panel
from rich.progress import TimeRemainingColumn, Task, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, \
    TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.style import Style
from rich.table import Column
from rich.text import Text

from .config import ConsoleConfig, ConsoleMode
from .utils import apply_style, calc_color_gradient
from .dataclasses import ContentItem, ListItems, ListItem
from .themes import OLDarkTheme
from .boxes import (
    EMPTY_BOX,
    TOP_BORDER,
    BOTTOM_BORDER,
    TOP_BOTTOM_BORDER,
    BOTTOM_PADDED_BORDER,
    TOP_PADDED_BORDER,
    TOP_BOTTOM_PADDED_BORDER,
)

class ConditionalTimeRemainingColumn(TimeRemainingColumn):
    """Renders time remaining estimate, but only for tasks where it makes sense."""

    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        is_application_task = task.fields.get('is_app_task', False)

        if is_application_task == "False" or is_application_task == "false":
            is_application_task = False

        if is_application_task and task.total is not None:
            # Display step count for the application task
            step_count = f"{int(task.completed)}/{int(task.total)} steps"
            return Text(step_count, style="progress.remaining")  # Use same style or a custom one
        else:
            if is_application_task:
                return Text("--:--:--", style="progress.remaining")
            else:
                return super().render(task)


class OLConsole:
    """
    The `OLConsole` class manages a configurable console display system with various modes such as
    null, logging, live, and normal. It establishes terminal interaction, logging, and visual
    customizations such as themes and fonts, allowing for versatile console-based application output.

    The console implements a singleton design pattern to ensure only one instance is maintained
    throughout the application. You can initialize the instance with a `ConsoleConfig`, which defines
    the behavior and mode of the console. Different modes (e.g., NULL, LOGGING, LIVE, NORMAL) are
    supported, each catering to specific use cases like silent operation, log-only operation, or full
    terminal interactions. The initialization method manages settings such as themes, colors, fonts,
    live displays, and logging to files. It also guarantees re-initialization with updated configurations
    or default settings. Designed for effectiveness and user flexibility, slight misconfigurations (such
    as missing parameters for LOGGING mode) can still be gracefully handled via warnings or exceptions.

    While the console provides robust support for output, improper use of certain modes (like missing
    `log_file` for LOGGING or invalid console modes) could lead to exceptions. Always ensure the `cfg`
    object is populated appropriately for intended behavior. Additionally, themes, fonts, or resources
    that fail to load might generate warnings or errors, ensuring clarity for debugging but requiring
    correct dependencies to avoid runtime issues.

    :ivar _instance: Singleton instance of the `OLConsole` class.
    :vartype _instance: OLConsole
    :ivar _console: The console object for handling terminal operations or output.
    :vartype _console: Console | None
    :ivar _cfg: Configuration for the console behavior and attributes.
    :vartype _cfg: ConsoleConfig | None
    :ivar _live: Instance managing live output (e.g., real-time UI or progress bars).
    :vartype _live: Live | None
    :ivar _log_file_handle: Opened file handle for logging mode, if applicable.
    :vartype _log_file_handle: Any | None
    :ivar _mode: The operating mode for the console (e.g., NULL, LOGGING, LIVE, NORMAL).
    :vartype _mode: ConsoleMode | None
    :ivar _progress_bar: Progress bar instance if applicable for the mode.
    :vartype _progress_bar: Progress | None
    :ivar _layout: The main layout structure of the console.
    :vartype _layout: Layout | None
    :ivar _title_content_layout: Layout for the console title content.
    :vartype _title_content_layout: Layout | None
    :ivar _main_content_layout: Layout for the main content.
    :vartype _main_content_layout: Layout | None
    :ivar _progress_content_layout: Layout for the progress content.
    :vartype _progress_content_layout: Layout | None
    :ivar _stats: A dictionary for managing stats or metrics to be displayed in the UI.
    :vartype _stats: dict
    :ivar _stats_display_layout: Layout for displaying stats-related content.
    :vartype _stats_display_layout: Layout | None
    :ivar _progress_bar_layout: Layout for the progress bar component.
    :vartype _progress_bar_layout: Layout | None
    :ivar _alternate_screens: List of alternative screen layouts for dynamic UI uses.
    :vartype _alternate_screens: list
    :ivar _alternate_screen_items: Items for representing content within alternate screens.
    :vartype _alternate_screen_items: list
    :ivar _main_content_cols: List of column layouts for the main content body.
    :vartype _main_content_cols: list of Layout
    :ivar _main_content_cols_panels: List of panels corresponding to the main content columns.
    :vartype _main_content_cols_panels: list of Panel
    :ivar _app_title: Default text for the application title display.
    :vartype _app_title: str
    :ivar _app_subtitle: Default text for the application subtitle display.
    :vartype _app_subtitle: str
    :ivar _app_stage: Text that indicates the current stage of the console application.
    :vartype _app_stage: str
    :ivar _stats_title: Title text for the stats section, customizable for different themes.
    :vartype _stats_title: Text
    :ivar _content_items: List of content items managed by the console.
    :vartype _content_items: list of ContentItem
    :ivar _tz_info: Timezone information for timestamped displays, customizable in config.
    :vartype _tz_info: ZoneInfo | None
    :ivar _main_content_panel: Panel containing the main content display.
    :vartype _main_content_panel: Panel | None
    :ivar _title_fig: Text-based figure associated with the application.
    :vartype _title_fig: Text | None
    :ivar _figlet_fonts: Preloaded figlet fonts for text-based visual outputs.
    :vartype _figlet_fonts: dict
    """
    _instance = None
    _console: Console|None = None
    _cfg: ConsoleConfig|None = None
    _live: Live|None = None
    _log_file_handle: Any|None = None
    _mode: ConsoleMode|None = None
    _progress_bar: Progress|None = None
    _layout: Layout|None = None
    _progress_tasks = {}

    # Layout content
    _title_content_layout: Layout | None = None
    _main_content_layout: Layout | None = None
    _progress_content_layout: Layout | None = None
    _stats = {}
    _stats_display_layout: Layout | None = None
    _stats_layout_width: int = 0
    _stats_layout_count: int = 0
    _stats_blocks: List[Layout] = []
    _stat_block_width: int = 25
    _stats_per_row: int = 0
    _alternate_screens = []
    _alternate_screen_items = []
    _main_content_cols: List[Layout] = []
    _main_content_cols_panels: List[Panel] = []

    # Default text content
    _app_title = "Ordered Learning"
    _app_subtitle = "Training"
    _app_stage = "Bootstrapping"
    _stats_title: Text = Text("")
    _content_items: List[ContentItem] = []
    _tz_info: ZoneInfo|None = None
    _main_content_panel: Panel|None = None
    _title_fig: Text|None = None
    _is_in_shutdown: bool = False

    _figlet_fonts: Dict[str, Figlet] = {}

    def __new__(cls, cfg: ConsoleConfig|None = None):
        """
        Creates and manages a singleton instance of the class, ensuring a single global
        instance with an option to reinitialize configuration.

        This block of code implements the singleton design pattern by overriding the
        __new__ method. It ensures that only one instance of the class is ever created.
        If the instance does not exist, it initializes a new one. If the instance
        already exists, it will optionally reinitialize the configuration if a different
        configuration is supplied. This ensures consistency by warning when trying to
        reinitialize the singleton with a different configuration after it has already
        been initialized. The pattern helps avoid creating multiple instances of a class,
        ensuring a single point of access for the instance.

        Special care should be taken to provide a consistent configuration since
        passing a new configuration to an already-instantiated object will result in
        reinitialization accompanied by a warning. This could be used incorrectly if
        users fail to ensure proper configuration before first-time initialization.

        :param cfg: Optional configuration object (ConsoleConfig). If None is
            provided, the method uses default parameters for initialization.

        :returns: Reference to the globally unique singleton instance with its
            configuration applied or re-applied.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(cfg)
        elif cfg is not None and cls._instance._cfg != cfg:
            cls._instance.print_warning("OLConsole already initialized with a different config. Re-initializing with new config.")
            cls._instance._initialize(cfg)
        return cls._instance

    def _initialize(self, cfg: ConsoleConfig|None = None):
        """
        Initializes the console configuration and sets up logging, display modes, and terminal styling.

        This method re-initializes the console object for different usage scenarios provided
        in the configuration. It supports multiple output modes such as logging, live display,
        and standard terminal output. Depending on the mode, it either sets up a terminal
        console object with optional colorization and styling or configures logging functionality
        to an external file. It also initializes resources such as figlet fonts when applicable.

        The function ensures proper cleanup and resets internal attributes to default values
        to avoid stale states during re-initialization. Be cautious when providing a configuration,
        as certain required attributes, such as `log_file` in LOGGING mode, must be properly
        defined. Misconfigurations may lead to runtime errors or warnings.

        Special Notes:
        - Be mindful of the `ConsoleMode` selected. Using unsupported modes raises a `ValueError`.
        - Ensure any provided `ConsoleConfig` instance has correct attributes, especially in
          modes like LOGGING where parameters such as `log_file` are mandatory.

        :param cfg: An instance of `ConsoleConfig` providing settings for mode, colors, timezone,
            and logging. Defaults to `None`, in which case a default `ConsoleConfig` object is used.
        :type cfg: ConsoleConfig | None
        :raises ValueError: When using LOGGING mode without a `log_file` or providing an unsupported
            console mode.
        :raises RuntimeError: When unable to open the specified log file in LOGGING mode.
        :return: None
        :rtype: None
        """
        # Close existing log file if re-initializing
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None

        if cfg is None:
            # Default configuration if none provided
            # This behavior might need adjustment based on project structure
            print("Warning: OLConsole initialized without explicit config.", file=sys.stderr)
            self._cfg = ConsoleConfig()
        else:
            self._cfg = cfg

        self._mode = self._cfg.mode
        self._tz_info = ZoneInfo(self._cfg.timezone) if self._cfg.timezone else None

        if not self._cfg.use_colors:
            self._cfg.color_system = None

        theme = OLDarkTheme()

        if self._mode == ConsoleMode.NULL:
            # Null mode: No display
            self._console = Console(
                quiet=True
            )
            self._live = None
            return

        elif self._mode == ConsoleMode.LOGGING:
            if not self._cfg.log_file:
                raise ValueError("log_file must be specified in ConsoleConfig for logging mode")
            try:
                # Open file in append mode, create if it doesn't exist
                self._log_file_handle = open(self._cfg.log_file, "a+", encoding="utf-8")
                self._console = Console(
                    file=self._log_file_handle,
                    theme=theme,
                    force_terminal=False,
                    no_color=True,
                )
            except IOError as e:
                raise RuntimeError(f"Failed to open log file {self._cfg.log_file}: {e}") from e
            self._live = None  # No live display in logging mode

        elif self._mode in (ConsoleMode.LIVE, ConsoleMode.NORMAL, ConsoleMode.SILENT):
            # Initialize console for terminal output
            no_color = not self._cfg.use_colors or self._cfg.color_system is None
            style = Style(bgcolor=OLDarkTheme.BACKGROUND) if not no_color else None
            self._console = Console(
                theme=theme,
                force_terminal=True if self._cfg.use_colors else None,
                force_jupyter=False,
                no_color=no_color,
                color_system=self._cfg.color_system.value if self._cfg.color_system else None,
                highlight=False,
                style=style
            )

            if self._mode == ConsoleMode.LIVE:
                self.start_live()

        else:
            raise ValueError(f"Unsupported console mode: {self._mode}")

        # Common initialization for non-null modes (if any)
        # Load figlet font etc. if needed and console exists
        if self._should_do_terminal():
            try:
                self._figlet_fonts["contessa"] = Figlet(font="contessa")
                self._figlet_fonts["cybermedium"] = Figlet(font="cybermedium")
                self._figlet_fonts["cybersmall"] = Figlet(font="cybersmall")
                self._figlet_fonts["univers"] = Figlet(font="univers")
            except Exception as e:
                if self._console:
                    self._console.print(f"[danger]Error loading resources: {e}[/danger]")
                else:
                    print(f"Error loading resources: {e}", file=sys.stderr)

    def _should_do_terminal(self):
        """
        Determines whether terminal interaction should occur based on the current
        console mode configuration.

        The method evaluates the mode attribute of the `_mode` value to derive if
        terminal-based operations are appropriate. Specific modes may explicitly
        disable or enable terminal interaction, ensuring the behavior aligns with
        the set console mode.

        :return: Boolean value indicating whether terminal interaction should be
            conducted.
        :rtype: bool
        """
        if self._mode == ConsoleMode.NULL:
            return False
        elif self._mode == ConsoleMode.LOGGING:
            return False
        elif self._mode == ConsoleMode.LIVE:
            return True
        elif self._mode == ConsoleMode.NORMAL:
            return True
        elif self._mode == ConsoleMode.SILENT:
            return True
        else:
            return True

    def _should_do_print(self):
        """
        Determines whether printing should be executed based on the current console
        mode. Returns False for NULL (no output at all) and SILENT (progress bar
        only), True for all other modes.

        :return: A boolean indicating whether printing should be executed or not
        :rtype: bool
        """
        if self._mode in (ConsoleMode.NULL, ConsoleMode.SILENT):
            return False
        else:
            return True

    def _should_do_ui(self):
        """
        Determines whether the UI should be displayed based on the current console mode.

        This method evaluates the current mode of the console, and if it is set to LIVE,
        it returns True indicating that UI-related actions should proceed. For all other
        modes, it will return False, meaning UI actions should not be performed.

        :return: A boolean indicating whether UI should be displayed.
        :rtype: bool
        """
        if self._mode == ConsoleMode.LIVE and self._live is not None:
            return True
        else:
            return False

    def progress_start(self):
        if not self._should_do_terminal():
            return
        self._create_progress_bar()
        if not self._should_do_ui():
            self._progress_bar.start()

    def progress_stop(self):
        if not self._should_do_terminal() or self._progress_bar is None:
            return
        self._progress_bar.stop()
        if self._should_do_ui() and not self._is_in_shutdown:
            self._layout["progress"].visible = False
            self._update_main_render()
        self._progress_bar = None
        self._progress_tasks = {}

    def start_live(self):
        """
        Initializes, configures, and starts the live display interface for the application's UI.

        This method sets up and begins a `Live` interface, which creates a dynamic real-time display
        within the console. It configures the layout of various UI components such as a custom title,
        progress section, main content area, and optional statistics display. Care is taken to
        ensure layouts are adaptive and crafted to fit the expected console structure. The method
        establishes these elements in a hierarchical layout structure using the `rich` library.

        Additionally, the live display is initialized only when certain conditions are met:
        the UI is enabled (`_should_do_ui()` is `True`), the live display is not yet running,
        and the console is properly initialized. Special gradients and styles are applied to the
        title and statistics sections for enhanced visual aesthetics, which are derived from
        theme configurations. The statistics display, if enabled in the configuration, has considerations
        for adaptive rendering to dynamically fit smaller or reshaped console windows in the future,
        though some aspects may require further refinement, as noted in the TODO comments.

        **Special Notes:**
          - Initialization is skipped if `_console` is `None`, `_live` already exists,
            or `_should_do_ui()` evaluates to `False`.
          - As noted in the code's TODO comments, the current system for displaying statistics
            using a simple `Text` object may require future redesign to ensure better alignment
            and adaptability to console resizing events.
          - The refresh rate for the live display is set to 10 frames per second, and output redirection
            (stdout and stderr) is disabled.

        :param self: The instance of the class initializing the live display.

        :raises RuntimeError: If one of the layouts or live interface fails to initialize properly
            due to unforeseen issues during live creation or layout assignment.

        :return: None
        """
        if self._mode == ConsoleMode.LIVE and self._live is None and self._console is not None:
            # Initialize panels for live display
            self._main_content_panel = Panel(
                renderable="",
                expand=True, border_style=OLDarkTheme.PURPLE, box=TOP_BOTTOM_BORDER, padding=0
            )
            # Set default width so get_display_width() works even without columns
            self._main_content_panel.width = self._console.width

            # Initialize content for live display
            title_lines = Figlet(font='contessa').renderText("OrderedLearning").splitlines()
            gradient = calc_color_gradient(OLDarkTheme.GRADIENT_BEGIN, OLDarkTheme.GRADIENT_END, len(title_lines))
            for i, line in enumerate(title_lines):
                title_lines[i] = apply_style(line, gradient[i])
            self._title_fig = Text.from_markup("\n".join(title_lines))
            # UI Layouts
            ## Base content section layouts
            self._title_content_layout = Layout(self._title_fig, name="title", size=3)
            self._main_content_layout = Layout(self._main_content_panel, name="main")
            self._progress_content_layout = Layout(name="progress", size=4)

            # Core screen layout
            self._layout = Layout()
            self._layout.split_column(
                self._title_content_layout,
                self._main_content_layout,
                self._progress_content_layout,
            )

            # Config progress section
            self._layout["progress"].visible = False

            # Define and start the Live object
            self._live = Live(
                self._layout,
                console=self._console,
                screen=True,
                refresh_per_second=10,
                redirect_stderr=False,
                redirect_stdout=False
            )
            self._live.start(refresh=True)
            self._cfg.use_live_display = True

    def end_live(self, set_transient: bool|None = None):
        """
        Stop and clean up the live status display and associated progress bar.

        This method ensures that the live status display is properly stopped and all
        related resources, such as the progress bar and live object, are cleaned up.
        If an exception occurs during the stopping process, a warning message is
        printed using the available console logger or the standard error stream.

        :raises Exception: If stopping the live object encounters an error,
            it will catch the exception and display an appropriate warning
            message.
        """
        try:
            self._is_in_shutdown = True
            self.progress_stop()
        except Exception as e:
            tb = e.__traceback__
            frames = traceback.extract_tb(tb)
            msg = f"Error stopping progress bar: {e}\n"
            msg += f"Frame count: {len(frames)}\n"
            msg += f"First five frames:\n"
            for i in range(5):
                msg += f"Frame {i}:\n"
                msg += f"File: {frames[i].filename}\n"
                msg += f"Line Number: {frames[i].lineno}\n"
                msg += f"Line: {frames[i].line}\n"
            raise RuntimeError(msg)
        finally:
            self._progress_bar = None
        if self._live is not None:
            try:
                if set_transient is not None:
                    self._live.transient = set_transient
                self._live.stop()
            except Exception as e:
                raise RuntimeError(f"Error stopping live: {e}")
            finally:
                self._live = None

    def handle_exception(self, show_locals: bool = False):
        self._console.print_exception(show_locals=show_locals)

    def print(self, content: str|ContentItem|RenderableType = "", style: str|Style = ""):
        """
        Handles printing of styled content with time information.

        This method encapsulates a given content along with its style and a timestamp
        into a `ContentItem` object, and subsequently processes it for display or
        logging using the private `_print_message` method. It simplifies content creation
        and supports dynamic styling by delegating styling logic to the `ContentItem`
        constructor.

        For Rich renderable objects (Panel, Table, Tree, etc.), the content is wrapped
        in a ContentItem with the appropriate renderable type and printed directly
        without text processing.

        Special Note:
            - The `ContentItem` class is expected to adhere to a specific data structure
              for proper functioning.
            - This method is dependent on the `_print_message` private method to handle
              the actual message display or processing.

        :param content: Content to be printed. Can be a string, a pre-existing
            `ContentItem`, or a Rich renderable (Panel, Table, Tree, etc.).
        :param style: Optional styling instructions for the content. Can either
            be a string or an object of type `Style`. Ignored for Rich renderables.
        :return: None
        """
        # Check if content is already a ContentItem
        if isinstance(content, ContentItem):
            self._print_message(content)
            return

        # Check if content is a Rich renderable (has __rich_console__ or __rich__ method)
        if hasattr(content, '__rich_console__') or hasattr(content, '__rich__'):
            # Determine the specific renderable type for better categorization
            content_type = "renderable"
            type_name = type(content).__name__.lower()
            if type_name == "panel":
                content_type = "panel"
            elif type_name == "table":
                content_type = "table"
            elif type_name == "tree":
                content_type = "tree"

            content_item = ContentItem(
                type=content_type,
                content=content,
                time=time.time()
            )
            self._print_message(content_item)
            return

        # Default: treat as text content
        content_item = ContentItem(
            type="text",
            content=content,
            style=style,
            time=time.time()
        )
        self._print_message(content_item)

    def print_list_item(self, list_item: ListItem, returns: bool = False):
        """
        Prints or returns a string representation of a list item.

        This function either prints a message containing a string representation of a
        list item or returns it as a string, based on the value of the `returns`
        parameter. If `returns` is True, the string representation of `list_item`
        is returned. Otherwise, a message containing the string representation and
        related metadata is printed.

        :param list_item: The list item to be converted into a string representation.
        :type list_item: ListItem
        :param returns: Determines whether the string representation of the list
            item is returned or printed. If True, the representation is returned
            instead of being printed. Defaults to False.
        :type returns: bool
        :return: If `returns` is True, returns the string representation of the
            list item. Otherwise, returns None.
        :rtype: str or None
        """
        if returns:
            return str(list_item)
        else:
            self._print_message(ContentItem(
                type="list_item",
                content=str(list_item),
                time=time.time(),
            ))
            return None

    def print_list(self, items: ListItems, returns: bool = False):
        """
        Prints a list of items or returns its string representation based on the specified behavior.

        This method converts the provided items into a string representation. If the `returns`
        parameter is set to True, the string representation of the items is returned. Otherwise,
        the method constructs a `ContentItem` object with the string representation of the items
        and prints it without returning any value.

        :param items: The list of items to be processed and converted to a string representation.
        :type items: ListItems
        :param returns: Determines whether the function should return the string representation
                        of the items (if True) or print the content using a `ContentItem` (if False).
                        Default is False.
        :type returns: bool
        :return: The string representation of the items if `returns` is True. Otherwise, returns None.
        :rtype: Optional[str]
        """
        output = str(items)
        if returns:
            return output
        else:
            self._print_message(ContentItem(
                type="list",
                content=output,
                time=time.time(),
            ))
            return None

    def print_notification(self, content: str):
        """
        Prints a notification message.

        This method creates a content item object specifically for a notification
        type message with the given content and the current timestamp. It then
        utilizes the `_print_message` method to process and display the constructed
        content item.

        :param content: The message content to be displayed in the notification.
        :type content: str
        """
        content_item = ContentItem(
            type="notification",
            content=content,
            time=time.time()
        )
        self._print_message(content_item)

    def print_warning(self, content: str):
        """
        Prints a warning message by creating a content item of type warning.

        This method prepares a content item using the provided warning content,
        associates it with the current timestamp, categorizes it as type "warning",
        and sends it for printing using an internal message handling function.

        :param content: The warning message content to be included in the content item.
        :type content: str
        :return: None
        """
        content_item = ContentItem(
            type="warning",
            content=content,
            time=time.time()
        )
        self._print_message(content_item)

    def print_error(self, content: str):
        """
        Logs an error message to the console using the ContentItem structure and the
        internal message printing mechanism.

        :param content: Message text to be logged as an error.
        :type content: str
        """
        content_item = ContentItem(
            type="error",
            content=content,
            time=time.time()
        )
        self._print_message(content_item)

    def print_complete(self, content: str):
        """
        Prints a complete message based on the provided content.

        This function takes a string input and formats it as a "complete" message
        using a `ContentItem` object. The formatted message includes the type,
        content provided, and the current time when the function is executed.
        Once formatted, the message is processed by an internal method
        to handle message printing.

        :param content: The text content that needs to be formatted and printed.
        :type content: str
        """
        content_item = ContentItem(
            type="complete",
            content=content,
            time=time.time()
        )
        self._print_message(content_item)

    def print_success(self, content: str):
        """
        Prints a success message encapsulated in a content item object.

        This method creates a `ContentItem` object with a type of "success"
        and the given content. It also records the current time of message
        creation. The content item is then passed to the `_print_message`
        method to handle further processing or display.

        :param content: The message content to be encapsulated and processed.
        :type content: str
        """
        content_item = ContentItem(
            type="success",
            content=content,
            time=time.time()
        )
        self._print_message(content_item)

    def rule(self, content, style: str|Style = ""):
        """
        Adds a rule with the specified content and style to the display system. Handles
        two modes of operation based on whether live display is enabled. When live
        display is active, the rule content is appended to the list of content items
        and the main rendering is updated. Otherwise, the rule is printed to the console
        immediately with the given style.

        :param content: The content to be displayed as a rule. This is the main
            text to be styled and added to the display.
        :type content: str
        :param style: The optional style to apply to the rule content. Can be
            either a string or a `Style` instance. An empty string may be
            provided as a default.
        :type style: str | Style
        :return: None
        """
        if not self._should_do_print():
            return
        if self._should_do_ui():
            self._content_items.append(ContentItem(
                type="rule",
                content=content,
                style=style
            ))
            self._update_main_render()
        else:
            self._console.rule(f"{apply_style(content, 'rule.text')}", style=style)

    def subrule(self, content, style: str|Style = ""):
        """
        Inserts a subrule content item with a specific style into the content list or
        renders it on the console, based on the current rendering mode.

        If the UI rendering mode is active, the content item is added to the `_content_items`
        list with a subrule type, specified content, and style. The main rendering process
        is subsequently updated to reflect the change. Alternatively, if the UI rendering
        mode is inactive, the content is printed to the console using the specified
        styling.

        :param content: The content of the subrule to add or render.
        :param style: Specific style to apply for the subrule content. Defaults to an
            empty string if no style is provided.
        :type style: str | Style
        :return: None
        """
        if not self._should_do_print():
            return
        if self._should_do_ui():
            self._content_items.append(ContentItem(
                type="subrule",
                content=content,
                style=style
            ))
            self._update_main_render()
        else:
            self._console.print(f"{apply_style(content, 'subrule.text')}", style=style)

    def section(self, content: str):
        """
        Adds a new section content item and updates the UI or prints it to the console.

        A section content item with the specified content is created and either appended
        to the UI rendering list or printed to the console, depending on whether UI
        updates are enabled.

        :param content: The text content for the section to be added.
        :type content: str
        """
        if not self._should_do_print():
            return
        section = ContentItem(
            type="section",
            content=content,
        )
        if self._should_do_ui():
            self._content_items.append(section)
            self._update_main_render()
        else:
            self._console.print(Align.center(Text.from_markup(str(section))))

    def clear_main_content(self):
        """
        Clears the main content of the UI or console output depending on the UI mode.

        This method is responsible for clearing the main content managed by the
        object. If the interface mode indicates that UI updates should occur,
        the internal content items list is reset, and the main rendering process
        is updated accordingly. Otherwise, it performs a clear action on a console
        interface.

        :raises RuntimeError: If an error occurs during UI update or console clear.

        :rtype: None
        """
        if self._should_do_ui():
            self._content_items = []
            self._update_main_render()
        else:
            self._console.clear()

    def new_main_content(
            self,
            content: str|RenderableType|ContentItem|List[ContentItem]|None = None,
            with_progress: bool = True
    ):
        """
        Updates the main content of the UI or console with the provided content. Depending on the
        type of the `content` parameter, the function processes the content to update the main
        panel, content items, or reset the renderable display to an empty state. The method also
        adjusts visibility of the progress section when applicable. If the UI mode is active,
        it maintains a record of alternate screens for navigation purposes.

        :param content: The content to update the main display with. This can be one of the
            following types:
            - str: A string of text, where each line will be split and added individually
                as `ContentItem` objects.
            - RenderableType: A renderable object such as a panel or other UI element.
            - ContentItem: A single content item to be appended to the content list.
            - List[ContentItem]: A list of content items to be appended.
            - None: Resets the main content to an empty state.
        :param with_progress: Indicates whether the progress section should be visible during
            the update. Defaults to True.
        :return: The index of the alternate screen that was updated, or None if not in UI mode.
        :rtype: Optional[int]
        """
        if self._should_do_ui():
            self._alternate_screens.append(self._main_content_panel.renderable)
            self._alternate_screen_items.append(self._content_items)
            self._layout["progress"].visible = with_progress
            self.clear_main_content()
            if isinstance(content, str):
                for line in content.splitlines():
                    self._content_items.append(ContentItem(
                        type="text",
                        content=line,
                    ))
            elif isinstance(content, RenderableType):
                self._main_content_panel.renderable = content
            elif isinstance(content, ContentItem):
                self._content_items.append(content)
            elif isinstance(content, list):
                for item in content:
                    self._content_items.append(item)
            else:
                self._main_content_panel.renderable = Text("")
            self._update_main_render()
            screen_idx = len(self._alternate_screens) - 1
            return screen_idx
        else:
            self._console.clear()
            return None

    def restore_main_content(self, screen_idx: int = -1):
        """
        Restores the main content of the application from a list of alternate screens.

        This function retrieves and restores the content and associated renderable items
        for the primary application view from stored alternate screens and items. It updates
        the main content panel to display the selected screen and adjusts its associated data
        items accordingly. This is particularly useful in scenarios where alternate screens
        have been stacked or managed temporally, allowing the user to revert back
        to a specified screen within the stack.

        The implementation ensures that the main content panel's renderable is updated with
        the correct alternate screen and synchronously adjusts the content items to reflect
        the screen's corresponding data. Caution must be exercised with the `screen_idx`
        parameter, as providing an out-of-range index can lead to unintended behavior if
        the `self._alternate_screens` or `self._alternate_screen_items` lists are
        not correctly managed or have mismatched lengths.

        :param screen_idx:
            An integer index indicating which alternate screen to restore. Defaults to `-1`,
            which pops the last screen in the stack. Must refer to a valid index within
            `self._alternate_screens` and `self._alternate_screen_items` lists.

        :raises IndexError:
            If `screen_idx` refers to an invalid index in `self._alternate_screens`
            or `self._alternate_screen_items`.

        :return:
            None
        """
        if self._should_do_ui() and len(self._alternate_screens) > 0 and len(self._alternate_screen_items) > 0:
            self._main_content_panel.renderable = self._alternate_screens.pop(screen_idx)
            self._content_items = self._alternate_screen_items.pop(screen_idx)
            self._update_main_render()

    def update_column_content(self, col_idx: int, content: str|RenderableType|None = None):
        """
        Update the content of a specific column in the console dynamically.

        This method is designed to update the renderable content of a specified column
        in a live console mode. It replaces the content of the column at the given
        index with new content, which can either be a string, a `RenderableType`, or
        `None`. If the provided content is a string, it will process the string line by
        line and convert it to a renderable representation. It ensures that only
        pre-existing columns corresponding to the given index are updated to avoid
        exceptions. This function only operates in `LIVE` console mode.

        .. note::
            - The method requires the console to be in `LIVE` mode for it to work. If the
              console is not in `LIVE` mode, no update will take place.
            - The column index provided (`col_idx`) should exist in the
              `_main_content_cols_panels` attribute. If the index is out of range, the
              method will skip the update without raising an exception.
            - For strings, the content is split into individual lines, and each line is
              processed using `Text.from_markup` to convert it into a `Renderable` object.
            - For `RenderableType` inputs, it directly updates the column with the `Renderable`.

        .. note::
            - This method will not create new columns if the specified index exceeds the
              current number of columns. Ensure the index corresponds to an existing column.
            - Improper use of markup in strings may result in rendering errors.

        :param col_idx: The index of the column to be updated. It must correspond to an
            existing column in `_main_content_cols_panels`.
            :type col_idx: int
        :param content: The new content for the column. It can be a string (parsed into
            renderable lines), a `RenderableType`, or `None` (to clear the column).
            :type content: str | RenderableType | None
        :return: None
        :raises TypeError: Raised if `content` is not of type `str`,
            `RenderableType`, or `None`.
        """
        if self._should_do_ui() and len(self._main_content_cols_panels) > col_idx:
            if isinstance(content, str):
                lines = [Text.from_markup(line) for line in content.splitlines()]
                self._main_content_cols_panels[col_idx].renderable = Group(*lines)
            elif isinstance(content, RenderableType):
                self._main_content_cols_panels[col_idx].renderable = content

    def add_column_to_main(self, width: int = 25, content: RenderableType|None = None):
        """
        Add a new column to the main content layout with configurable width and optional content.
        This method dynamically updates the main content layout in a live display setup. If the
        live display is disabled, no columns will be added, and the method will return `None`.
        The columns created are updated and maintained within internal data structures to ensure
        proper rendering in the layout configuration.

        :param width: The width of the new column in units. Defaults to 25 if not specified.
        :type width: int
        :param content: Optional content to be added to the new column. If not provided, a
            default `Panel` with predefined properties will be created and used as content.
        :type content: RenderableType | None
        :return: The index of the newly created column in the list of main content columns if
            the live display is active. Returns `None` if the live display is disabled.
        :rtype: int | None
        """
        if self._should_do_ui():
            col_idx = len(self._main_content_cols)
            new_col_name = f"col_{col_idx}"
            layout = Layout(name=new_col_name, size=width)
            if content is None:
                content = Panel(
                    "",
                    expand=True, border_style=OLDarkTheme.MAGENTA, box=SQUARE, padding=0
                )
            layout.update(content)
            if len(self._main_content_layout.children) == 0:
                sub_main_layout = Layout(self._main_content_panel, name="sub_main")
                self._main_content_layout.split_row(sub_main_layout, layout)
            else:
                self._main_content_layout.add_split(layout)
            self._main_content_cols.append(layout)
            self._main_content_cols_panels.append(content)
            self._main_content_panel.width = self._console.width - sum([layout.size for layout in self._main_content_cols])
            self._update_main_render()
            return col_idx
        else:
            return None

    def remove_columns_from_main(self):
        """
        Removes all columns from the main content area and updates the associated layout and render.

        This method is primarily used to reset or clear the display of the main content area by removing
        any existing columns and reverting the layout to its initial state. It checks whether certain UI
        conditions are met and ensures that the process only occurs if there are columns currently
        present in the main content area. By resetting the `self._main_content_cols` list, undoing any
        splitting in the layout, and reapplying the main content panel settings, the method guarantees
        a consistent layout state across subsequent updates. This operation is safe to use but requires
        careful attention in scenarios where columns may be inadvertently removed.

        :return: None
        """
        if self._should_do_ui() and len(self._main_content_cols) > 0:
            self._main_content_cols = []
            self._main_content_layout.unsplit()
            self._main_content_layout.update(self._main_content_panel)
            self._update_main_render()

    def update_app_title(self, title: str):
        self._app_title = title

    def update_app_subtitle(self, subtitle: str):
        self._app_subtitle = subtitle

    def update_app_stage(self, stage: str):
        self._app_stage = stage

    def update_app_stats(self, stats: dict):
        """Update application stats dict. No-op for display (stats panel removed).

        Kept for API compatibility — callers can still push stats without error.
        """
        self._stats.update(stats)

    def _calculate_stats_layout(self) -> tuple[int, int, int]:
        """Calculate stats layout dimensions based on console width.

        Returns:
            Tuple of (stats_per_row, total_rows, block_width)
        """
        available_width = self._console.width - 4  # Account for borders
        if available_width <= 0:
            available_width = 76  # Fallback for edge cases

        stats_per_row = max(1, available_width // self._stat_block_width)
        total_stats = len(self._stats)
        total_rows = min(2, (total_stats + stats_per_row - 1) // stats_per_row) if total_stats > 0 else 0
        actual_block_width = available_width // stats_per_row if stats_per_row > 0 else available_width
        return stats_per_row, total_rows, actual_block_width

    def _render_stat_block(self, name: str, value: str, width: int) -> Text:
        """Render a single stat block with truncation if needed.

        :param name: The stat name to display.
        :param value: The stat value to display.
        :param width: The maximum width allocated for this stat block.
        :return: A styled Text object containing the formatted stat.
        """
        separator = ": "
        value_str = str(value)
        separator_len = len(separator)

        # Calculate available space for name and value
        name_max = (width - separator_len) // 2
        value_max = width - separator_len - min(len(name), name_max)

        # Truncate with ~ if needed
        if len(name) > name_max:
            display_name = name[:name_max - 1] + "~"
        else:
            display_name = name

        if len(value_str) > value_max:
            display_value = value_str[:value_max - 1] + "~"
        else:
            display_value = value_str

        result = Text()
        result.append(display_name, style="stats.name")
        result.append(separator, style="stats.name")
        result.append(display_value, style="stats.value")
        return result

    def _rebuild_stats_layout(self):
        """Rebuild the stats layout structure when dimensions change.

        Creates a grid of stat blocks organized into rows, with each block
        allocated equal space within the available console width.
        """
        self._stats_layout_width = self._console.width
        self._stats_layout_count = len(self._stats)
        self._stats_blocks = []

        stats_per_row, total_rows, block_width = self._calculate_stats_layout()
        self._stats_per_row = stats_per_row

        # Build the layout structure
        # Title row + content rows
        title_layout = Layout(self._stats_title, name="stats_title", size=1)

        if total_rows == 0:
            # No stats, just show title
            self._stats_display_layout.split_column(title_layout)
            return

        # Create row layouts
        row_layouts = []
        for row_idx in range(total_rows):
            row_layout = Layout(name=f"stats_row_{row_idx}", size=1)
            block_layouts = []

            for col_idx in range(stats_per_row):
                stat_idx = row_idx * stats_per_row + col_idx
                block_layout = Layout(name=f"stat_block_{stat_idx}", ratio=1)
                block_layouts.append(block_layout)
                self._stats_blocks.append(block_layout)

            row_layout.split_row(*block_layouts)
            row_layouts.append(row_layout)

        # Combine title and rows
        self._stats_display_layout.split_column(title_layout, *row_layouts)

    def _update_stats_values(self):
        """Update stat block values without rebuilding the layout structure.

        Iterates through the stats and updates each corresponding block
        with the rendered and centered stat content.
        """
        if not self._stats_blocks:
            return

        _, _, block_width = self._calculate_stats_layout()
        stat_items = list(self._stats.items())
        max_displayable = len(self._stats_blocks)

        for idx, block in enumerate(self._stats_blocks):
            if idx < len(stat_items):
                name, value = stat_items[idx]
                rendered = self._render_stat_block(name, str(value), block_width)
                block.update(Align.center(rendered))
            else:
                # Empty block for unfilled slots
                block.update(Text(""))

    def create_progress_task(self, task_name: str, task_desc: str, total: float|None = None, is_app_task: bool = False, **kwargs):
        """
        Creates a task in the progress bar and tracks its status.

        This function is responsible for initializing a new task to monitor using
        the progress bar. If the progress bar has not been started, it is initialized
        automatically. Each task is identified by a unique name provided as input,
        and additional metadata about the task, such as its description and total
        steps, are stored for tracking purposes.

        :param task_name: The unique name used to identify the progress task.
        :type task_name: str
        :param task_desc: A short description providing details about the progress task.
        :type task_desc: str
        :param total: The total number of steps to complete the task. If None, an
            indeterminate progress task is created.
        :type total: float | None, optional
        :param is_app_task: Indicates whether the task is part of an application-level
            job.
        :type is_app_task: bool, optional
        :param kwargs: Additional keyword arguments passed to customize task creation.
            These may include task-specific configuration.
        :return: None
        """
        if self._should_do_terminal():
            if self._progress_bar is None:
                self.progress_start()
            task_id = self._progress_bar.add_task(task_desc, total=total, is_app_task=is_app_task, **kwargs)
            self._progress_tasks[task_name] = {"id": task_id, "total": total, "description": task_desc, "completed": 0}

    def update_progress_task(self, task_name: str, completed: float|None = None, **kwargs):
        """
        Updates the progress of a specific task within the progress management system.
        This function allows updating the `completed` value of a task directly or by
        incrementing it using the `advance` value specified in `kwargs`. It ensures that
        the progress tasks and progress bar are synchronized, and removes the task if it
        is completed and there are other tasks in progress.

        :param task_name: The name of the task to update.
        :type task_name: str
        :param completed: The new completed value for the task. If None, the existing
                          value is retained or updated using the 'advance' parameter
                          in kwargs.
        :type completed: float | None
        :param kwargs: Additional parameters to update the task. This may include
                       'advance' to increment the completed value or other custom
                       parameters.
        :return: True if the task was successfully updated, False if the task was not found.
        :rtype: bool
        """
        if self._should_do_terminal():
            task_config = self._get_progress_task(task_name)
            if task_config is None:
                return False
            updates = {"completed": task_config["completed"] + kwargs["advance"] if "advance" in kwargs else 0}
            updates["completed"] = completed if completed is not None else updates["completed"]
            updates.update(kwargs)
            self._progress_tasks[task_name].update(updates)
            self._progress_bar.update(task_config["id"], completed=completed, **kwargs)
            target_task = None
            for task in self._progress_bar.tasks:  # Iterate through the current tasks list
                if task.id == task_config["id"]:
                    target_task = task
                    break  # Found the task
            if target_task is not None and target_task.finished and len(self._progress_tasks) > 1:
                self.remove_progress_task(task_name)
            return True
        else:
            return False

    def remove_progress_task(self, task_name: str):
        """
        Removes a specified progress task from the progress bar if it exists and updates its state appropriately.

        This method checks if terminal operations are allowed (`_should_do_terminal`) and whether the given
        task name exists in the current progress tasks. If found, it retrieves the task configuration,
        updates the progress bar if the task has a total defined, stops the task, and removes it entirely
        from the progress bar. Finally, it deletes the task from the internal `_progress_tasks` dictionary
        and returns a Boolean indicating the successful removal. If the task does not exist or terminal
        operations are not enabled, it returns `False`.

        Care must be taken to ensure the `task_name` exists in `_progress_tasks` before calling this method.
        Additionally, the method assumes that the internal structure and integrity of `_progress_tasks` and
        related configurations (like `task_config`) are properly managed and not tampered with. Failure to
        adhere to this could result in unexpected behavior.

        Special Notes:
        - This method depends on `_should_do_terminal()` to check if terminal operations are active.
        - The format and structure of `task_config` returned by `_get_progress_task()` are critical to the
          operation of this method. Ensure that this underlying method is correctly implemented.

        :param task_name: The name of the progress task to be removed.
        :type task_name: str

        :return: A Boolean indicating whether the task was successfully removed.
        :rtype: bool
        """
        if self._should_do_terminal() and task_name in self._progress_tasks.keys():
            task_config = self._get_progress_task(task_name)
            if task_config is None:
                return False
            if task_config["total"] is not None:
                self._progress_bar.update(task_config["id"], completed=task_config["total"])
            self._progress_bar.stop_task(task_config["id"])
            self._progress_bar.remove_task(task_config["id"])
            del self._progress_tasks[task_name]
            return True
        else:
            return False

    def reset_progress_task(self, task_name: str, total: float | None = None, description: str | None = None):
        """
        Reset a progress task to start fresh, optionally with a new total and description.

        This method resets the completed count to 0 and optionally updates the total
        and description. Useful for restarting a task without removing and recreating it.

        :param task_name: The name of the task to reset.
        :type task_name: str
        :param total: New total for the task. If None, keeps the existing total.
        :type total: float | None
        :param description: New description for the task. If None, keeps the existing description.
        :type description: str | None
        :return: True if the task was successfully reset, False if the task was not found.
        :rtype: bool
        """
        if self._should_do_terminal():
            task_config = self._get_progress_task(task_name)
            if task_config is None:
                return False

            # Update internal tracking
            self._progress_tasks[task_name]["completed"] = 0
            if total is not None:
                self._progress_tasks[task_name]["total"] = total
            if description is not None:
                self._progress_tasks[task_name]["description"] = description

            # Reset the underlying progress bar task
            update_kwargs = {"completed": 0}
            if total is not None:
                update_kwargs["total"] = total
            if description is not None:
                update_kwargs["description"] = description

            self._progress_bar.reset(task_config["id"], **update_kwargs)
            return True
        else:
            return False

    def get_progress_task_properties(self, task_name: str) -> dict|None:
        """
        Retrieves the progress task properties for a given task name.

        This function determines whether progress task properties should be retrieved
        by first checking if a terminal-related condition is met using the method
        `_should_do_terminal()`. If the condition holds true, it fetches the
        configuration for the specified task using an internal method `_get_progress_task()`.
        If no configuration is found, it explicitly returns `None`. Otherwise, it returns
        the configuration dictionary. If the terminal-related condition is not met, it
        returns `None` immediately.

        Special care should be taken with how this function is used, particularly when
        processing its return value. Callers should be prepared to handle cases where
        `None` is returned, which could either indicate that the terminal condition was
        not satisfied or that the task configuration was absent. Errors can occur if
        the return value is assumed to be a dictionary without proper checks.

        :param task_name: The name of the task whose progress properties are to be retrieved.
        :type task_name: str
        :return: A dictionary containing the task properties if found and the terminal
            condition is satisfied; None otherwise.
        :rtype: dict | None
        """
        if self._should_do_terminal():
            task_config = self._get_progress_task(task_name)
            if task_config is None:
                return None
            return task_config
        else:
            return None

    def has_progress_task(self, task_name: str):
        """
        Checks if a task with the given name exists in the current progress tasks.

        This method verifies whether a particular task, identified by its name, is
        present in the dictionary containing ongoing progress tasks. The function
        utilizes the `in` keyword to check if the `task_name` exists within the
        keys of the internal `_progress_tasks` dictionary. It assumes that
        `_progress_tasks` is a dictionary where task names serve as keys and the
        associated values represent task information.

        **Note:** Be cautious while passing `task_name`. If the dictionary key
        lookup fails due to incorrect task names, the function will return `False`
        instead of throwing exceptions. Always ensure that `task_name` matches
        exactly as keys in `_progress_tasks`.

        :param task_name: The name of the task to check for its existence.
        :type task_name: str
        :return: A boolean indicating whether the specified task exists in the
                 progress tasks dictionary.
        :rtype: bool
        """
        return task_name in self._progress_tasks.keys()

    def prompt(self, prompt_text: str, default_value: str|None = None, password: bool = False, choices: List[str]|None = None):
        """
        Prompts the user for input through the appropriate interface based on configuration.

        This function provides an interactive prompt to the user, either via a UI-based approach using a custom
        rendering engine or through a terminal-based fallback (e.g., `Prompt.ask`). It supports customizable
        prompt messages, default values, password inputs, and a set of choices for the user to select from.
        The function handles special keys for navigation, deletion, and clipboard pasting while ensuring
        a user-friendly and interactive experience.

        The implementation determines whether to use the UI-based interface or the terminal-based fallback
        based on the methods `_should_do_ui` and `_should_do_terminal`. When using the UI approach,
        the key events (such as arrow keys and Enter) are handled manually to enable real-time input
        feedback and navigation through the available choices. In the terminal-based fallback, the
        `rich.prompt.Prompt.ask` handles the interaction. An incorrect implementation or improper input
        values can lead to inconsistencies (e.g., empty choices or invalid default values).

        .. note::
            - Ensure that `self._stats_display_layout` and `self._live` are properly initialized for
              the UI-based flow to render correctly.
            - The function gracefully falls back to terminal input when UI interaction is not configured.
            - The function will return `None` when no input mechanisms are available.

        .. warning::
            - Ensure `choices` is a list of strings for compatibility with the UI-based and terminal-based
              prompt flows.
            - Do not supply contradictory inputs, such as setting `password=True` while also expecting
              visible feedback.

        .. tip::
            - Use `CTRL+V` to paste content directly into the input buffer during the UI-based
              interaction.
            - Provide clear and concise prompt text to enhance usability in both UI-based and
              terminal-based flows.
            - Use arrow keys for cycling through choices when applicable.

        :param prompt_text: The text to display as the prompt message.
        :type prompt_text: str

        :param default_value: An optional default value to return if the user provides no input or presses
            `ESC`. Defaults to `None`.
        :type default_value: str | None

        :param password: If `True`, hides the user input (e.g., for passwords) by displaying
            asterisks `*`. Defaults to `False`.
        :type password: bool

        :param choices: A list of options from which the user can select. When present, arrow keys are
            enabled for cycling through options. Defaults to `None`.
        :type choices: List[str] | None

        :return: The user-provided input as a string. Returns the default value if the user presses `ESC` or
            enters no input. Returns `None` if no input mechanism is available.
        :rtype: str | None

        :raises TypeError: Directly raised if the `choices` parameter is not a list or contains
            non-string elements.
        """
        if self._should_do_ui() and self._stats_display_layout is not None:
            _prompt_text = f"{apply_style('OL', 'orange')}{apply_style(':', 'purple')} {apply_style(prompt_text.strip(), 'prompt.instruction')}"
            _choices_text = f"{apply_style('[', 'purple')} "
            for choice in choices:
                if choice != choices[0]:
                    _choices_text += ", "
                if choice == default_value:
                    _choices_text += f"{apply_style(choice, 'prompt.default')} {apply_style('(default)', 'purple')}"
                else:
                    _choices_text += f"{apply_style(choice, 'prompt.choices')}"
            if default_value is not None:
                _choices_text += f" {apply_style('] (ESC for default, UP and DOWN to cycle choices)', 'purple')}"
            else:
                _choices_text += f"{apply_style('] (UP and DOWN to cycle choices)', 'purple')}"

            _input_buffer = []
            _output_text = _prompt_text + "\n" + _choices_text + "\n" + apply_style('> ', 'magenta')
            _selected_choice = None
            _choice_index = 0

            _stats_panel_content = self._stats_display_layout.renderable if self._stats_display_layout is not None else Text("")

            while True:
                # This blocks the main thread until a key is pressed
                k = readkey()

                # We only want to handle a few special keys
                if k == key.ENTER:
                    _selected_choice = "".join(_input_buffer)
                    break
                elif k == key.BACKSPACE or k == key.DELETE:
                    if len(_input_buffer) > 0:
                        _input_buffer.pop()
                elif k == key.ESC:
                    _selected_choice = default_value
                    break
                elif k == key.UP or k == key.LEFT:
                    if not choices or len(choices) == 0:
                        continue
                    _input_buffer = list(choices[_choice_index])
                    if _choice_index == 0:
                        _choice_index = len(choices) - 1
                    else:
                        _choice_index -= 1
                elif k == key.DOWN or k == key.RIGHT:
                    if not choices or len(choices) == 0:
                        continue
                    _input_buffer = list(choices[_choice_index])
                    if _choice_index == len(choices) - 1:
                        _choice_index = 0
                    else:
                        _choice_index += 1
                elif k == key.CTRL_V:
                    _input_buffer = list(pyperclip.paste())
                elif k.isprintable():
                    _input_buffer.append(k)

                if not password:
                    self._stats_display_layout.update(Text.from_markup(_output_text + apply_style("".join(_input_buffer), 'text')))
                else:
                    self._stats_display_layout.update(Text.from_markup(_output_text + apply_style("*" * len(_input_buffer), 'text')))

                self._live.refresh()

            self._stats_display_layout.update(_stats_panel_content)
            self._live.refresh()

            if _selected_choice is None:
                _selected_choice = default_value
            return _selected_choice
        elif self._should_do_terminal():
            prompt_input = Prompt.ask(
                prompt=prompt_text,
                console=self._console,
                default=default_value,
                password=password,
                choices=choices
            )
            return prompt_input
        else:
            return None

    def confirm(self, prompt_text: str, default_value: bool = False):
        """
        Handles user confirmation prompts in different console modes.

        This method provides an interactive confirmation prompt to the user, with visual
        and keyboard event-based handling in `LIVE` mode and a simpler, blocking mechanism
        in `NORMAL` mode. It uses styled output to enhance readability and provides
        sensible defaults, which are visually highlighted. The method is designed to execute
        differently depending on the mode of operation (`LIVE` or `NORMAL`), offering
        flexibility to adapt to varying runtime contexts. In `LIVE` mode, the function
        responds dynamically to user keystrokes, while in `NORMAL` mode, it leverages
        blocking confirmation through `Confirm.ask`. If called in other modes, it
        gracefully returns `None`.

        .. note::
            Ensure that the `_mode` is either `LIVE` or `NORMAL` before invoking this
            method to avoid unexpected `None` return values.

        :param prompt_text: The message text to display in the prompt.
        :type prompt_text: str
        :param default_value: The default confirmation choice (`True` for "Yes",
            `False` for "No"). Defaults to `False`.
        :type default_value: bool
        :return: The user’s response to the confirmation prompt, as `True` or `False`,
            or `None` if the console mode is unsupported.
        :rtype: Optional[bool]
        """
        if self._should_do_ui() and self._stats_display_layout is not None:
            _prompt_text = f"{apply_style('OL', 'orange')}{apply_style(':', 'purple')} {apply_style(prompt_text.strip(), 'prompt.instruction')}"
            _yes_text = f"{apply_style('Y', 'prompt.default')}" if default_value else f"{apply_style('y', 'prompt.choices')}"
            _no_text = f"{apply_style('N', 'prompt.default')}" if not default_value else f"{apply_style('n', 'prompt.choices')}"
            _default_text = apply_style('[' + _yes_text + '/' + _no_text + ']', 'purple')
            _output_text = _prompt_text + "\n" + _default_text + "\n" + apply_style('> ', 'magenta')

            _stats_panel_content = self._stats_display_layout.renderable if self._stats_display_layout is not None else Text("")
            _selected_choice = default_value

            self._stats_display_layout.update(Text.from_markup(_output_text))
            self._live.refresh()

            while True:
                # This blocks the main thread until a key is pressed
                k = readkey()

                # We only want to handle a few special keys
                if k == key.ENTER or k == key.ESC:
                    break
                elif k == 'y' or k == 'Y':
                    _selected_choice = True
                    break
                elif k == 'n' or k == 'N':
                    _selected_choice = False
                    break

            self._stats_display_layout.update(_stats_panel_content)
            self._live.refresh()

            return _selected_choice
        elif self._should_do_terminal():
            confirm_input = Confirm.ask(
                prompt=prompt_text,
                console=self._console,
                default=default_value
            )
            return confirm_input
        else:
            return None

    def measure(self, content: str) -> Measurement:
        """
        Measure the width required to display a given string in the console.

        This method determines the amount of horizontal space the input string
        will occupy when rendered in the console. It utilizes the internal
        console object's `measure` method to perform this calculation, which
        may take into consideration factors such as text styling, formatting,
        and any additional rendering logic specific to the console
        implementation.

        This method requires a valid string input and operates under the
        assumption that the console object is properly initialized and
        capable of processing the measurement operation. To avoid unexpected
        behavior, ensure that the provided string (`content`) is suitable for
        rendering in the console environment.

        **Special Notes:**
          - The console's specific rendering and measurement logic may affect
            the result. If accurate width measurement is critical, confirm
            alignment between the string formatting and the console setup.

        :param content: The string content to be measured for display width
            in the console.
        :type content: str
        :raises AttributeError: If the internal `_console` object does not
            support a `measure` method or is not properly initialized.
        :return: A new `Measurement` instance with the width of the string
            in pixels as a `minimum` and `maximum` value.
        :rtype: Measurement
        """
        return self._console.measure(content)

    def get_time_format(self):
        return self._cfg.time_format

    def get_display_width(self, output_location: Literal["main", "column"] = "main", column: int = -1) -> int:
        if output_location == "main":
            # Handle NORMAL mode where layouts aren't initialized
            if self._main_content_layout is None:
                return self._console.width if self._console else 80
            if len(self._main_content_layout.children) > 0:
                if self._main_content_layout.get("sub_main").size is not None:
                    message_line_width = self._main_content_layout.get("sub_main").size
                else:
                    message_line_width = self._console.width - sum([layout.size for layout in self._main_content_cols])
            else:
                message_line_width = self._main_content_panel.width if self._main_content_panel else self._console.width
        else:
            message_line_width = self.get_column_width(column)
            if message_line_width is None:
                raise ValueError(f"Cannot determine width of invalid or uninitialized column index: {column}")

        return message_line_width

    def get_column_width(self, col_idx: int = -1):
        if len(self._main_content_cols) > 0:
            return self._main_content_cols[col_idx].size
        else:
            return None

    @property
    def is_live(self) -> bool:
        """Whether the console is in LIVE display mode."""
        return self._cfg.use_live_display if self._cfg else False

    def get_console_config(self) -> ConsoleConfig:
        return self._cfg

    def get_tz_info(self) -> ZoneInfo|None:
        return self._tz_info

    def _get_progress_task(self, task_name: str):
        if self.has_progress_task(task_name):
            return self._progress_tasks[task_name]
        else:
            return None

    def _print_message(self, text: str | ContentItem, with_time=True):
        if not self._should_do_print():
            return
        if self._should_do_ui():
            if isinstance(text, str):
                text = ContentItem(
                    type="text",
                    content=text,
                    time=time.time() if with_time else None
                )
            self._content_items.append(text)
            self._update_main_render()
        else:
            # For renderable types, print the raw renderable directly
            if isinstance(text, ContentItem) and text.is_renderable:
                self._console.print(text.renderable)
            else:
                self._console.print(str(text))

    def _update_main_render(self):
        if self._cfg.use_live_display:
            message_line_count = self._console.height
            if self._layout['progress'].visible:
                message_line_count -= 4 # Progress panel
            message_line_count -= 3  # Title bar
            message_line_count -= 2  # Top and bottom border
            message_line_width = self.get_display_width()
            message_line_width -= 2 # Left and right border
            content_items = self._content_items.copy()
            content_items.reverse()
            displayed_items = []
            line_count = 0
            for item in content_items:
                lines = item.splitlines()
                lines.reverse()
                for line in lines:
                    if line_count > message_line_count:
                        break

                    if item.type == "section" or item.type == "rule" or item.type == "subrule":
                        line = line.center()

                    if isinstance(line, Text):
                        rich_line = line
                    else:
                        rich_line = Text.from_markup(str(line))

                    if len(rich_line) > message_line_width:
                        sublines = rich_line.wrap(
                                console=self._console,
                                width=message_line_width
                        )
                        sublines = [*sublines]
                        sublines.reverse()
                        line_count += len(sublines)
                        displayed_items.extend(sublines)
                    else:
                        line_count += 1
                        displayed_items.append(line)
            displayed_items.reverse()
            renderables = displayed_items[-message_line_count:]
            self._main_content_panel.renderable = Group(*renderables)
            self._live.refresh()

    def _create_progress_bar(self):
        spinner_col = SpinnerColumn(table_column=Column(max_width=3))
        desc_col = TextColumn(text_format="[progress.description]{task.description}", style='progress.description', table_column=Column(max_width=30, min_width=15))
        bar_col = BarColumn(bar_width=None)
        pct_col = TaskProgressColumn(table_column=Column(max_width=10))
        time_elapsed_col = TimeElapsedColumn(table_column=Column(max_width=15))
        time_remaining_col = ConditionalTimeRemainingColumn(table_column=Column(max_width=15))
        self._progress_bar = Progress(
            spinner_col, desc_col, bar_col, pct_col, time_elapsed_col, time_remaining_col,
            console=self._console, transient=True, expand=True
        )

        if self._cfg.use_live_display and self._layout is not None:
            self._progress_content_layout.update(self._progress_bar)
            self._layout["progress"].visible = True
            self._update_main_render()
