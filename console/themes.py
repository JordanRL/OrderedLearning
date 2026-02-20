from rich.style import Style
from rich.theme import Theme


class OLDarkTheme(Theme):
    """
    Represents a dark theme configuration for a user interface.

    This class defines a consistent color palette and style settings for
    various visual components, including text, icons, notifications,
    headers, lists, progress bars, and other elements commonly used
    in user interface design. The theme is specifically crafted to
    provide a visually appealing dark mode experience with harmonious
    gradients, clear contrasts, and vibrant highlights.

    :ivar BLUE: Light blue color used for gradients and various styles.
    :type BLUE: str
    :ivar RICH_BLUE: Rich and vibrant blue color.
    :type RICH_BLUE: str
    :ivar CYAN: Cyan color for specific style purposes.
    :type CYAN: str
    :ivar GREEN: Green color used for success indicators.
    :type GREEN: str
    :ivar YELLOW: Yellow color for warnings and separators.
    :type YELLOW: str
    :ivar RED: Red color used for error indicators.
    :type RED: str
    :ivar ORANGE: Orange color used for stylistic elements like rules.
    :type ORANGE: str
    :ivar MED_GREY: Medium grey for subtle text and icons.
    :type MED_GREY: str
    :ivar PURPLE: Primary purple color for headers and accents.
    :type PURPLE: str
    :ivar DARK_PURPLE: Darker purple for sub-rules and accents.
    :type DARK_PURPLE: str
    :ivar LAVENDER: Lavender color used for highlighted headers.
    :type LAVENDER: str
    :ivar MAGENTA: Vibrant magenta as a gradient end for highlights.
    :type MAGENTA: str
    :ivar PINK: Bright pink for progress indicators and accents.
    :type PINK: str
    :ivar RICH_PINK: A rich pink used for progress spinners and extras.
    :type RICH_PINK: str
    :ivar BACKGROUND: Base background color for the theme.
    :type BACKGROUND: str
    :ivar DEFAULT_TEXT: Default text color for general content.
    :type DEFAULT_TEXT: str
    """
    BLUE = GRADIENT_BEGIN = '#61AFEF'
    RICH_BLUE = '#4B6BFF'
    CYAN = '#56B6C2'
    GREEN = '#98C379'
    YELLOW = '#E5C07B'
    RED = '#E06C75'
    ORANGE = '#D19A66'
    MED_GREY = '#8A8F98'
    PURPLE = '#663399'
    DARK_PURPLE = '#4B0082'
    LAVENDER = '#B87FD9'
    MAGENTA = GRADIENT_END = '#BE50AE'
    PINK = '#FF69B4'
    RICH_PINK = '#FF1493'
    BACKGROUND = '#282C34'
    DEFAULT_TEXT = '#F8E8EC'

    def __init__(self):
        super().__init__({
            # Basic colors
            "blue": Style(color=self.BLUE),
            "rich_blue": Style(color=self.RICH_BLUE),
            "cyan": Style(color=self.CYAN),
            "green": Style(color=self.GREEN),
            "yellow": Style(color=self.YELLOW),
            "red": Style(color=self.RED),
            "orange": Style(color=self.ORANGE),
            "med_grey": Style(color=self.MED_GREY),
            "purple": Style(color=self.PURPLE),
            "dark_purple": Style(color=self.DARK_PURPLE),
            "lavender": Style(color=self.LAVENDER),
            "magenta": Style(color=self.MAGENTA),
            "pink": Style(color=self.PINK),
            "rich_pink": Style(color=self.RICH_PINK),

            # General color styles
            "header": Style(color=self.PURPLE),
            "default": Style(color=self.DEFAULT_TEXT),
            "text": Style(color=self.DEFAULT_TEXT),

            # Content type styles
            "notification.icon": Style(color=self.PURPLE),
            "notification.content": Style(color=self.BLUE),
            "complete.icon": Style(color=self.GREEN),
            "complete.content": Style(color=self.BLUE),
            "warning.icon": Style(color=self.ORANGE),
            "warning.content": Style(color=self.YELLOW),
            "error.icon": Style(color=self.RED),
            "error.content": Style(color=self.RED),
            "info.icon": Style(color=self.BLUE),
            "info.content": Style(color=self.DEFAULT_TEXT),

            # Header types
            "rule.text": Style(color=self.ORANGE),
            "rule.line": Style(color=self.BLUE),
            "rule.arrow_left": Style(color=self.BLUE),
            "rule.arrow_right": Style(color=self.BLUE),
            "subrule.text": Style(color=self.RICH_BLUE),
            "subrule.arrow_left": Style(color=self.DARK_PURPLE),
            "subrule.arrow_right": Style(color=self.DARK_PURPLE),

            # Time display
            "time.numbers": Style(color=self.ORANGE),
            "time.separator": Style(color=self.YELLOW),
            "time.ampm": Style(color=self.YELLOW),
            "time.brackets": Style(color=self.DARK_PURPLE),

            # List display
            "list.header": Style(color=self.DARK_PURPLE),
            "list.bullet": Style(color=self.PURPLE),
            "list.separator": Style(color=self.PURPLE),
            "list.title": Style(color=self.LAVENDER),
            "list.content": Style(color=self.BLUE),

            # Prompt
            "prompt.choices": Style(color=self.BLUE),
            "prompt.instruction": Style(color=self.PURPLE),

            # Progress
            "bar.complete": Style(color=self.RICH_BLUE),
            "bar.finished": Style(color=self.GREEN),
            "bar.pulse": Style(color=self.RICH_PINK),
            "progress.description": Style(color=self.RICH_BLUE),
            "progress.filesize": Style(color=self.GREEN),
            "progress.filesize.total": Style(color=self.GREEN),
            "progress.download": Style(color=self.GREEN),
            "progress.elapsed": Style(color=self.YELLOW),
            "progress.percentage": Style(color=self.RICH_BLUE),
            "progress.remaining": Style(color=self.PINK),
            "progress.data.speed": Style(color=self.GREEN),
            "progress.spinner": Style(color=self.RICH_PINK),
            "status.spinner": Style(color=self.RICH_PINK),

            # Stats display
            "stats.name": Style(color=self.LAVENDER),
            "stats.value": Style(color=self.CYAN),

            # Experiment semantic styles
            "metric.improved": Style(color=self.GREEN, bold=True),
            "metric.degraded": Style(color=self.RED, bold=True),
            "metric.value": Style(color=self.CYAN),
            "metric.label": Style(color=self.MED_GREY),
            "context": Style(color=self.MED_GREY),
            "strategy": Style(color=self.YELLOW),
            "target": Style(color=self.GREEN),
            "phase": Style(color=self.MAGENTA),
            "success": Style(color=self.GREEN, bold=True),

            # Semantic text roles (all muted, independently tunable)
            "label": Style(color=self.MED_GREY),
            "status": Style(color=self.MED_GREY),
            "detail": Style(color=self.MED_GREY),
            "description": Style(color=self.MED_GREY, italic=True),
            "placeholder": Style(color=self.MED_GREY),

            # Data display
            "path": Style(color=self.GREEN),
            "trigger": Style(color=self.CYAN),
            "value.count": Style(color=self.CYAN, bold=True),

            # Table
            "table.header": Style(color=self.MED_GREY, bold=True),

            # Panel borders
            "panel.primary": Style(color=self.MAGENTA),
            "panel.success": Style(color=self.GREEN),
            "panel.attention": Style(color=self.YELLOW),
            "panel.info": Style(color=self.BLUE),

            # Hook display
            "hook.name": Style(color=self.GREEN, bold=True),
            "hook.type.observer": Style(color=self.CYAN),
            "hook.type.intervention": Style(color=self.MAGENTA),

            # Accuracy scale
            "accuracy.excellent": Style(color=self.GREEN, bold=True),
            "accuracy.good": Style(color=self.YELLOW),
            "accuracy.fair": Style(color=self.CYAN),
            "accuracy.poor": Style(color=self.MED_GREY),

            # Misc
            "divider": Style(color=self.MED_GREY),
            "caption": Style(color=self.MED_GREY, italic=True),
        })