from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

PURPLE = "#a855f7"
PINK = "#f472b6"


def configure_plot_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111116",
            "axes.facecolor": "#111116",
            "text.color": "#e9d5ff",
            "axes.labelcolor": "#e9d5ff",
            "xtick.color": "#d8c8ed",
            "ytick.color": "#d8c8ed",
            "grid.alpha": 0.2,
            "font.size": 10,
        }
    )


def plot_category_frequencies(
    counts: dict[str, Counter[str]], reviews_per_category: int, top_n: int = 10
) -> Figure:
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, layout="constrained"
    )
    maximum = max(
        value
        for category in counts.values()
        for _, value in category.most_common(top_n)
    )
    for axis, category, title, color in zip(
        axes,
        ["neg", "pos"],
        ["Negative reviews", "Positive reviews"],
        [PURPLE, PINK],
        strict=True,
    ):
        items = counts[category].most_common(top_n)[::-1]
        words, frequencies = zip(*items)
        bars = axis.barh(words, frequencies, color=color)
        axis.bar_label(bars, padding=4, fontsize=9)
        axis.set(
            title=title,
            xlabel=f"Occurrences in {reviews_per_category} reviews",
            xlim=(0, maximum * 1.18),
        )
        axis.grid(axis="x")
        axis.set_axisbelow(True)
    figure.suptitle(f"Top {top_n} words after English stopword removal")
    return figure


def plot_category_comparison(comparison: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    positions = list(range(len(comparison)))
    for shift, column, label, color in [
        (-0.18, "negative share (%)", "Negative reviews", PURPLE),
        (0.18, "positive share (%)", "Positive reviews", PINK),
    ]:
        bars = axis.barh(
            [position + shift for position in positions],
            comparison[column],
            height=0.36,
            label=label,
            color=color,
        )
        axis.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    axis.set_yticks(positions, comparison["word"])
    axis.invert_yaxis()
    axis.set(
        title="The same words in negative and positive reviews",
        xlabel="Share of all remaining words in the category (%)",
    )
    axis.margins(x=0.25)
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    return figure
