"""Data-related display functions.

Content analysis and dataset statistics displays.
"""

from rich.table import Table
from rich.panel import Panel
from rich import box

from console import OLConsole


def display_content_analysis(console, buffer, terms, sample_size=10000):
    """Display how frequently analysis terms appear in the data pool buffer.

    Scans up to sample_size documents from the buffer and reports per-term
    match counts. Useful for understanding how much target-relevant content
    exists in the training data (e.g., how many Wikipedia articles mention
    "Kepler" or "astronomy").

    Args:
        console: OLConsole instance (may be None).
        buffer: List of text strings from the DataPool.
        terms: List of search terms to look for.
        sample_size: Maximum number of documents to scan.
    """
    if console is None:
        console = OLConsole()

    n = min(len(buffer), sample_size)
    if n == 0 or not terms:
        return

    # Count documents containing each term
    term_counts = {term: 0 for term in terms}
    any_term_count = 0

    for text in buffer[:n]:
        text_lower = text.lower()
        found_any = False
        for term in terms:
            if term.lower() in text_lower:
                term_counts[term] += 1
                found_any = True
        if found_any:
            any_term_count += 1

    # Build table
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        padding=(0, 2),
        title=f"[label]Content analysis[/label] [detail]({n:,} documents sampled)[/detail]",
    )
    table.add_column("Term", style="label")
    table.add_column("Documents", style="value.count", justify="right")
    table.add_column("Frequency", style="metric.value", justify="right")

    for term in terms:
        count = term_counts[term]
        pct = 100.0 * count / n if n > 0 else 0.0
        table.add_row(term, f"{count:,}", f"{pct:.2f}%")

    table.add_row(
        "[bold]Any term[/bold]",
        f"[bold]{any_term_count:,}[/bold]",
        f"[bold]{100.0 * any_term_count / n:.2f}%[/bold]",
    )

    console.print(table)
