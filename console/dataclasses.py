import datetime
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Literal, List, Tuple, Union, Iterable

from pyfiglet import Figlet
from rich._wrap import divide_line
from rich.console import RenderableType, ConsoleOptions, Console
from rich.errors import MarkupError
from rich.segment import Segment
from rich.style import Style
from rich.text import Text, Span
from rich.markup import _parse, Tag

from .config import TimeFormat
from .utils import apply_style, calc_color_gradient
from .themes import OLDarkTheme

SECTION_FONTS = [
    Figlet(font="cyberlarge"),
    Figlet(font="cybermedium"),
    Figlet(font="cybersmall"),
]

STANDARD_GRADIENTS = [
    calc_color_gradient(OLDarkTheme.GRADIENT_BEGIN, OLDarkTheme.GRADIENT_END, 3),
]

def _get_console():
    from .olconsole import OLConsole
    return OLConsole()

class PrintableItem:
    content: str
    _spans: List[Span]|None = None
    _rendered_content: RenderableType|None = None

    def __init__(self, content: str):
        displayed_text = ""
        styles: List[Tuple[int, Tag]] = []
        spans: List[Span] = []
        pop = styles.pop

        def pop_style(popped_style: str) -> Tuple[int, Tag]:
            """Pop tag matching given style name."""
            for index, (_, opened_tag) in enumerate(reversed(styles), 1):
                if opened_tag.name == popped_style:
                    return styles.pop(-index)
            raise KeyError(popped_style)

        for position, plain_text, tag in _parse(content):
            if plain_text is not None:
                plain_text = plain_text.replace("\\[", "[")
                displayed_text += plain_text
            elif tag is not None:
                if tag.name.startswith("/"):
                    style_name = tag.name[1:].strip()

                    if style_name:  # explicit close
                        style_name = Style.normalize(style_name)
                        try:
                            start, open_tag = pop_style(style_name)
                        except KeyError:
                            raise MarkupError(
                                f"closing tag '{tag.markup}' at position {position} doesn't match any open tag"
                            ) from None
                    else:  # implicit close
                        try:
                            start, open_tag = pop()
                        except IndexError:
                            raise MarkupError(
                                f"closing tag '[/]' at position {position} has nothing to close"
                            ) from None

                    spans.append(Span(start, len(displayed_text), str(open_tag)))
                else:
                    styles.append((len(displayed_text), Tag(Style.normalize(tag.name), tag.parameters)))

        text_length = len(displayed_text)
        while styles:
            start, tag = styles.pop()
            style = str(tag)
            if style:
                spans.append(Span(start, text_length, style))

        self._spans = sorted(spans[::-1], key=attrgetter("start"))

        self.content = displayed_text

    def _wrap(self, available_width: int) -> Tuple[int, str]:
        """
        Wraps the content based on available_width and returns a tuple that is in the format:

        (number_of_lines, wrapped_string_content)
        """
        raw_lines = self.content.splitlines()
        wrapped_lines = []

        for raw_line in raw_lines:
            offsets = divide_line(self.content, available_width)

        return len(wrapped_lines), "\n".join(wrapped_lines)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Iterable[Segment]:
        if self._rendered_content is None:
            output = Text.from_markup(self.__str__())
        else:
            output = self._rendered_content
        yield from console.render(output, options)

@dataclass
class ContentItem(PrintableItem):
    """
    Representation of a styled content item with various formats and layouts.

    This class encapsulates content and supports operations for styling, formatting,
    and layout in a text-based console or display. The class provides mechanisms
    to control the representation and manipulation of the content, including
    splitting, modification, and textual transformations. It also integrates
    styling logic for specific content types and flexible layout options.

    :ivar type: Specifies the type of the content (e.g., "text", "section", "notification").
    :ivar content: The textual content of the item, potentially pre-wrapped or processed.
    :ivar height: Computed height of the text in lines, determined by its length and the display space.
    :ivar style: Style applied to the content, can be a string, `Style` instance, or None.
    :ivar time: An optional timestamp or duration associated with the content.
    :ivar location: Specifies where the content should be displayed, defaults to "main".
    :ivar column: Column index for multi-column displays, defaults to -1.
    :ivar padding_top: Number of padding lines above the content, defaults to 0.
    :ivar padding_bottom: Number of padding lines below the content, defaults to 0.
    :ivar _font: Internal list of font configurations used for sections, defaults to SECTION_FONTS.
    :ivar _section_repr: Internal cached representation of the content section, defaults to None.
    :ivar _current_printable_space: Internal cached space available for printable content, defaults to None.
    :ivar _content_length: Cached length of the content, used in length calculations, defaults to None.
    :type type: Literal["text", "section", "rule", "subrule", "notification", "error", "success", "complete", "warning", "info", "list", "list_item"]
    :type content: str
    :type height: int
    :type style: str | Style | None
    :type time: float | None
    :type location: Literal["main", "column"]
    :type column: int
    :type padding_top: int
    :type padding_bottom: int
    :type _font: List[Figlet]
    :type _section_repr: List[str] | None
    :type _current_printable_space: int | None
    :type _content_length: int | None
    """
    type: Literal[
            "text",
            "rendered_text",
            "section",
            "rule",
            "subrule",
            "notification",
            "error",
            "success",
            "complete",
            "warning",
            "info",
            "list",
            "list_item",
            "panel",
            "table",
            "tree",
            "renderable",
        ]
    height: int
    style: str|Style|None = None
    time: float|None = None
    location: Literal["main", "column"] = "main"
    column: int = -1
    padding_top: int = 0
    padding_bottom: int = 0
    _font: Figlet|None = None
    _section_repr: str|None = None
    _current_printable_space: int|None = None
    _content_length: int|None = None
    _raw_renderable: RenderableType|None = None

    def __init__(
            self,
            type: Literal[
                "text",
                "rendered_text",
                "section",
                "rule",
                "subrule",
                "notification",
                "error",
                "success",
                "complete",
                "warning",
                "info",
                "list",
                "list_item",
                "panel",
                "table",
                "tree",
                "renderable",
            ],
            content: str | RenderableType,
            style: str|Style|None = None,
            time: float|None = None
    ):
        self.type = type
        self.style = style
        self.time = time

        # Handle Rich renderable types differently
        if type in ("panel", "table", "tree", "renderable"):
            self._raw_renderable = content
            self.content = ""  # Empty string for text-based operations
            self._rendered_content = content
            # Set defaults for fields used by _printable_space property
            self.location = "main"
            self.column = -1
            self._current_printable_space = None  # Will be set dynamically in _printable_space
            # Height and length will be calculated dynamically in splitlines/render
            self.height = 0
            self._content_length = 0
            return

        # Set content first - must be done before calling methods that depend on it
        self.content = content

        if self.style is None:
            self.style = ""
        elif isinstance(self.style, str):
            self.style = Style.parse(self.style)
        else:
            pass
        self._current_printable_space = _get_console().get_display_width(self.location, self.column)

        self._font = Figlet("cybermedium", width=self._current_printable_space-22)

        self._rendered_content = Text.from_markup(self.__str__())
        self._recalc()

    def __len__(self) -> int:
        if self._content_length is not None:
            return self._content_length
        return _get_console().measure(self._rendered_content).maximum

    @property
    def is_renderable(self) -> bool:
        """Check if this ContentItem contains a Rich renderable object."""
        return self.type in ("panel", "table", "tree", "renderable")

    @property
    def renderable(self) -> RenderableType | None:
        """Get the raw Rich renderable object, if this is a renderable type."""
        return self._raw_renderable if self.is_renderable else None

    def copy(self, content: str, as_line: bool = False) -> "ContentItem":
        if as_line:
            content = content.replace('\n', '')
            time = None
        else:
            time = self.time

        copied_item = ContentItem(
            type=self.type,
            content=content,
            style=self.style,
            time=time
        )

        return copied_item

    def line_count(self) -> int:
        return self.height

    def splitlines(self) -> List["ContentItem"]:
        # For renderables, render to ContentItem lines at current width
        if self.is_renderable and self._raw_renderable is not None:
            return self._render_to_lines()

        raw_lines = self.content.splitlines()
        lines = [self.copy(content, True) for content in raw_lines]

        return lines

    def _render_to_lines(self) -> List["ContentItem"]:
        """Render a Rich renderable to a list of ContentItem lines at current width.

        This method renders Panel, Table, Tree, or other Rich renderables
        into individual ContentItem lines that can be processed by the main renderer.
        The rendering uses the current printable space width, so it adapts
        to console resizing and column changes.

        :return: List of ContentItem objects, one per rendered line.
        """
        if self._raw_renderable is None:
            return []

        console = _get_console()
        width = self._printable_space

        # Fallback to console width if _printable_space returned None
        if width is None:
            width = console._console.width if console._console else 80

        # Remove margins
        width -= 2

        # Create console options with the current width
        options = console._console.options.copy()
        options.max_width = width

        # Render the renderable to lines of segments
        rendered_lines = list(console._console.render_lines(
            self._raw_renderable,
            options
        ))

        # Convert each line of segments to a ContentItem with pre-set _rendered_content
        content_items = []
        for segment_line in rendered_lines:
            text = Text()
            for segment in segment_line:
                if segment.text:
                    text.append(segment.text, style=segment.style)

            content_items.append(ContentItem(type="rendered_text", content=text.markup))

        return content_items

    def split(self, sep: str = None, maxsplit: int = -1) -> List["ContentItem"]:
        raw_lines = self.content.split(sep, maxsplit)
        lines = [self.copy(content, True) for content in raw_lines]

        return lines

    def _split_raw(self, sep: str = None, maxsplit: int = -1) -> str:
        pass

    def strip(self) -> "ContentItem":
        return self._unary_modify_and_recalc("strip")

    def lstrip(self) -> "ContentItem":
        return self._unary_modify_and_recalc("lstrip")

    def rstrip(self) -> "ContentItem":
        return self._unary_modify_and_recalc("rstrip")

    def replace(self, old: str, new: str, count: int = -1) -> "ContentItem":
        return self._map_modify_and_recalc(lambda x: x.replace(old, new, count))

    def lpad(self, width: int, fillchar: str = " ") -> "ContentItem":
        return self._map_modify_and_recalc(lambda x: fillchar * width + x)

    def rpad(self, width: int, fillchar: str = " ") -> "ContentItem":
        return self._map_modify_and_recalc(lambda x: x + fillchar * width)

    def center(self) -> "ContentItem":
        line_offset = (self._printable_space - self.__len__()) // 2
        return self._map_modify(lambda x: " " * line_offset + x)

    def __radd__(self, other: Union["ContentItem", str]) -> "ContentItem":
        return self._binary_modify(other, "radd")

    def __iadd__(self, other: Union["ContentItem", str]) -> "ContentItem":
        return self._binary_modify_and_recalc(other, "iadd")

    def __add__(self, other: Union["ContentItem", str]) -> "ContentItem":
        return self._binary_modify(other, "add")

    def _map_modify(self, func, copy: bool = True):
        if copy:
            self.copy(func(self.content))
        else:
            self.content = func(self.content)
            self._rendered_content = self._pre_wrapped_str(self.content)
        return self

    def _map_modify_and_recalc(self, func):
        self._map_modify(func, False)
        self._recalc()
        return self

    def _binary_modify(self, other: Union["ContentItem", str], op: Literal["add", "iadd", "radd"], copy: bool = True):
        new_content = self.content
        if isinstance(other, ContentItem):
            if op == "add":
                new_content += other.content
            elif op == "iadd":
                new_content += other.content
            elif op == "radd":
                new_content = other.content + self.content
            else:
                raise ValueError("Invalid operation.")
        else:
            if op == "add":
                new_content += other
            elif op == "iadd":
                new_content += other

        if copy:
            self.copy(new_content)
        else:
            self.content = new_content
            self._rendered_content = self._pre_wrapped_str(new_content)

        return self

    def _binary_modify_and_recalc(self, other: Union["ContentItem", str], op: Literal["add", "iadd", "radd"]):
        self._binary_modify(other, op, False)
        self._recalc()
        return self

    def _unary_modify(self, op: Literal["strip", "lstrip", "rstrip"], copy: bool = True):
        if op == "strip":
            new_content = self.content.strip()
        elif op == "lstrip":
            new_content = self.content.lstrip()
        elif op == "rstrip":
            new_content = self.content.rstrip()
        else:
            raise ValueError("Invalid operation.")

        if copy:
            self.copy(new_content)
        else:
            self.content = new_content
            self._rendered_content = self._pre_wrapped_str(new_content)

        return self

    def _unary_modify_and_recalc(self, op: Literal["strip", "lstrip", "rstrip"]):
        self._unary_modify(op, False)
        self._recalc()
        return self

    def _recalc(self):
        self._content_length = self.__len__()
        self.height = (self._content_length // self._printable_space) \
                      + (1 if self._content_length % self._printable_space != 0 else 0)

        return self

    @property
    def _printable_space(self):
        if self._current_printable_space != _get_console().get_display_width(self.location, self.column):
            self._current_printable_space = _get_console().get_display_width(self.location, self.column)

        return self._current_printable_space

    @staticmethod
    def _theme_wrap(text: str, style: str):
        return f"[{style}]{text}[/{style}]"

    def _text_str(self, content: str) -> str:
        match self.type:
            case "notification":
                content_style = "notification.content" if not self.style else self.style
                content = f"[notification.icon]ⓘ[/notification.icon] {apply_style(content, content_style)}"
            case "error":
                content_style = "error.content" if not self.style else self.style
                content = f"[error.icon]ⓧ[/error.icon] {apply_style(content, content_style)}"
            case "complete":
                content_style = "complete.content" if not self.style else self.style
                content = f"[complete.icon]✔[/complete.icon] {apply_style(content, content_style)}"
            case "warning":
                content_style = "warning.content" if not self.style else self.style
                content = f"[warning.icon]⚠[/warning.icon] {apply_style(content, content_style)}"
            case "info":
                content_style = "info.content" if not self.style else self.style
                content = f"[info.icon]ⓘ[/info.icon] {apply_style(content, content_style)}"
            case "success":
                content_style = "success" if not self.style else self.style
                content = f"{apply_style(content, content_style)}"
            case "text":
                content_style = "text" if not self.style else self.style
                content = f"{apply_style(content, content_style)}"
            case _:
                pass

        total_pad = 1
        if _get_console().get_console_config().show_time:
            if self.time is not None:
                dt_utc = datetime.datetime.fromtimestamp(self.time, tz=datetime.timezone.utc)
                dt_target_tz = dt_utc.astimezone(_get_console().get_tz_info())
                time_format = _get_console().get_console_config().time_format.value
                time_format = time_format.replace(
                    ":",
                    f"{ContentItem._theme_wrap(':', 'time.separator')}"
                ).replace(
                    "%p",
                    f"{ContentItem._theme_wrap('%p', 'time.ampm')}"
                )
                formatted_time_string = dt_target_tz.strftime(time_format)
                content = f"{ContentItem._theme_wrap('[', 'time.brackets')}" \
                          f"{ContentItem._theme_wrap(formatted_time_string, 'time.numbers')}" \
                          f"{ContentItem._theme_wrap(']', 'time.brackets')} {content}"
            else:
                match _get_console().get_console_config().time_format:
                    case TimeFormat.NO_SECONDS:
                        total_pad += 11
                    case TimeFormat.NO_AM_PM:
                        total_pad += 11
                    case TimeFormat.NO_SECONDS_NO_AM_PM:
                        total_pad += 8
                    case TimeFormat.TWENTY_FOUR_HOUR:
                        total_pad += 11
                    case TimeFormat.TWENTY_FOUR_HOUR_NO_SECONDS:
                        total_pad += 8
                    case _:
                        total_pad += 14
        content = " " * total_pad + content
        return content

    def _section_str(self, content: str) -> str:
        if self._section_repr is None:
            self._section_repr = self._font.renderText(self.content).strip()
        gradient = calc_color_gradient(OLDarkTheme.GRADIENT_BEGIN, OLDarkTheme.GRADIENT_END, 3)
        lines = self._section_repr.splitlines()
        self._section_repr = "\n".join([
            f"{ContentItem._theme_wrap(line, gradient[i])}"
            for i, line in enumerate(lines)
        ])
        self._section_repr += "\n"
        return self._section_repr

    def _rule_str(self, content: str) -> str:
        reserved_chars = len(content) + 4 + 30
        num_bars = (self._printable_space - reserved_chars) // 2

        new_content = ContentItem._theme_wrap("━"*num_bars + "▶ ", "rule.arrow_left") \
                      + ContentItem._theme_wrap(content, "rule.text") \
                      + ContentItem._theme_wrap(" ◀" + "━"*num_bars, "rule.arrow_right")
        return new_content

    def _subrule_str(self, content: str) -> str:
        reserved_chars = len(content) + 4 + 30
        num_bars = (self._printable_space - reserved_chars) // 2

        new_content = ContentItem._theme_wrap("─" * num_bars + "► ", "subrule.arrow_left") \
                      + ContentItem._theme_wrap(content, "subrule.text") \
                      + ContentItem._theme_wrap(" ◄" + "─" * num_bars, "subrule.arrow_right")
        return new_content

    def _pre_wrapped_str(self, content) -> str:
        match self.type:
            case "section":
                return self._section_str(content)
            case "rule":
                return self._rule_str(content)
            case "subrule":
                return self._subrule_str(content)
            case "rendered_text":
                return content
            case _:
                return self._text_str(content)

    def __str__(self) -> str:
        string_content = self._pre_wrapped_str(self.content)

        if self.style is not None:
            if isinstance(self.style, str) and self.style != "":
                string_content = ContentItem._theme_wrap(string_content, self.style)
            elif isinstance(self.style, Style):
                string_content = ContentItem._theme_wrap(string_content, self.style.__str__())
        else:
            pass

        return string_content

@dataclass
class ListItem(PrintableItem):
    """
    Represents an item in a list, including its title, content, bullet, and separator.

    This class is used to define and manipulate items in a formatted list. Each item
    contains a title, content, a bullet that appears before the title, and a separator
    character that is displayed between the title and its content. The separator can be
    customized as needed through a method. The formatted item can also be converted
    to its string representation, respecting any custom styling applied to the bullet,
    title, separator, and content.

    :ivar title: The title of the list item.
    :type title: str
    :ivar content: The content associated with the list item.
    :type content: str
    :ivar bullet: Symbol or string to act as the bullet for the list item.
        Defaults to "»".
    :type bullet: str
    :ivar separator: The separator symbol between the title and content.
        Defaults to ":".
    :type separator: str
    """
    title: str
    content: str
    bullet: str = '»'

    separator: str = ":"

    def change_separator(self, new_separator: str):
        self.separator = new_separator

    def __str__(self):
        text = f"{apply_style(self.bullet, 'list.bullet')} " if self.bullet != "" else ""
        text += f"{apply_style(self.title, 'list.title')}{apply_style(self.separator, 'list.separator')} {apply_style(self.content, 'list.content')}"
        return text

@dataclass
class ListItems(PrintableItem):
    """
    Represents a list that contains items and supports headers and formatting.

    The ListItems class is used to create and manage a list of items, with the
    flexibility to define the type of list (bullet or numbered) and an optional
    header. It handles list item formatting and string representation.

    :ivar items: The items contained in the list.
    :type items: List[ListItem]
    :ivar type: Specifies the type of list, either 'bullet' or 'numbered'.
    :type type: Literal["bullet", "numbered"]
    :ivar header: An optional header for the list, which can be a string,
        ContentItem, or None.
    :type header: str|ContentItem|None
    """
    items: List[ListItem]
    type: Literal["bullet", "numbered"] = "bullet"
    header: Union["ContentItem", str, None] = None

    @classmethod
    def make_from_list(
            cls,
            items: List[str|ContentItem|ListItem],
            header: Union["ContentItem", str, None] = None,
            type: Literal["bullet", "numbered"] = "bullet",
            bullet_char: str = '»',
    ):
        for i, item in enumerate(items):
            if isinstance(item, str):
                title, content = item.split(":", 1)
                items[i] = ListItem(bullet=bullet_char, title=title, content=content)
            elif isinstance(item, ContentItem):
                title, content = item.content.split(":", 1)
                items[i] = ListItem(bullet=bullet_char, title=title, content=content)
            elif isinstance(item, ListItem):
                items[i] = item
            else:
                raise ValueError("Invalid item type.")
        cls(items=items, header=header, type=type)

    def __str__(self):
        if self.header is None:
            header_str = ""
        elif isinstance(self.header, ContentItem):
            header_str = str(self.header)
        else:
            header_str = self.header

        text = ""
        if header_str != "":
            text += f"[list.header]{header_str}[/list.header]\n"

        for i, item in enumerate(self.items):
            if self.type == "bullet":
                item.bullet = '»'
            else:
                item.bullet = str(i+1) + "."
            text += f"{item}\n"

        return text