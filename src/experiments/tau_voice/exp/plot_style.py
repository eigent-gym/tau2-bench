"""
Shared plot styling utilities for tau_voice experiment analysis.

This module provides consistent colors, styles, and helper functions
for matplotlib plots across performance_analysis.py and voice_analysis.py.

LLM colors are deterministic based on the LLM name hash, ensuring the same
LLM always gets the same color regardless of load order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Configuration Constants
# =============================================================================

# Speech complexity levels in order
# Single-feature ablations first, then pairwise, then full
SPEECH_COMPLEXITIES = [
    "control",
    "control_audio",
    "control_accents",
    "control_behavior",
    "control_audio_accents",
    "control_audio_behavior",
    "control_accents_behavior",
    "regular",
]

# Domains in order
DOMAINS = ["retail", "airline", "telecom"]

# Domain colors - soft, harmonious palette
DOMAIN_COLORS = {
    "retail": "#7FBCD2",  # Soft sky blue
    "airline": "#E8A87C",  # Soft coral/salmon
    "telecom": "#8FBC8F",  # Soft sage green
}

# Speech complexity colors
SPEECH_COMPLEXITY_COLORS = {
    "control": "#76B7B2",  # Teal
    # Single-feature ablations
    "control_audio": "#59A14F",  # Green
    "control_accents": "#EDC948",  # Yellow
    "control_behavior": "#B07AA1",  # Purple
    # Pairwise ablations
    "control_audio_accents": "#4E79A7",  # Steel blue
    "control_audio_behavior": "#E15759",  # Coral red
    "control_accents_behavior": "#9C755F",  # Brown
    # Full
    "regular": "#F28E2B",  # Orange
}

# Speech complexity markers
SPEECH_COMPLEXITY_MARKERS = {
    "control": "o",
    # Single-feature ablations
    "control_audio": "^",
    "control_accents": "D",
    "control_behavior": "v",
    # Pairwise ablations
    "control_audio_accents": "<",
    "control_audio_behavior": ">",
    "control_accents_behavior": "p",
    # Full
    "regular": "s",
}

# Speech complexity display names for figures/tables
SPEECH_COMPLEXITY_DISPLAY_NAMES = {
    "control": "Control",
    # Single-feature ablations
    "control_audio": "w/ Noise",
    "control_accents": "w/ Accents",
    "control_behavior": "w/ Interrupts",
    # Pairwise ablations
    "control_audio_accents": "w/ Noise+Accents",
    "control_audio_behavior": "w/ Noise+Interrupts",
    "control_accents_behavior": "w/ Accents+Interrupts",
    # Full
    "regular": "Full",
}


def get_complexity_display_name(complexity: str) -> str:
    """Get display name for a complexity level."""
    return SPEECH_COMPLEXITY_DISPLAY_NAMES.get(complexity, complexity.capitalize())


# =============================================================================
# Provider / Model Mapping
# =============================================================================

# Provider display names (canonical mapping from provider key to human-readable name)
PROVIDER_DISPLAY = {
    "gemini": "Google",
    "openai": "OpenAI",
    "xai": "xAI",
    "amazon": "Amazon",
}

# Provider ordering for tables and plots (by provider key)
PROVIDER_ORDER_KEYS = ["gemini", "openai", "xai", "amazon"]


def get_provider_key(llm: str) -> str:
    """Map an LLM identifier to its provider key.

    Args:
        llm: Full LLM identifier (e.g., "openai:gpt-realtime-2025-08-28")

    Returns:
        Provider key string (e.g., "openai")
    """
    llm_lower = llm.lower()
    if "gpt" in llm_lower or "openai" in llm_lower:
        return "openai"
    elif "gemini" in llm_lower or "google" in llm_lower:
        return "gemini"
    elif "grok" in llm_lower or "xai" in llm_lower:
        return "xai"
    elif "nova" in llm_lower or "amazon" in llm_lower:
        return "amazon"
    else:
        return llm.split(":")[-1][:10].lower()


def get_provider_display(provider_key: str) -> str:
    """Get the human-readable display name for a provider key.

    Args:
        provider_key: Provider key (e.g., "openai")

    Returns:
        Display name (e.g., "OpenAI")
    """
    return PROVIDER_DISPLAY.get(provider_key.lower(), provider_key.capitalize())


def get_model_sort_key(llm: str) -> tuple:
    """Get a sort key for an LLM that orders by provider first, then model name.

    Args:
        llm: Full LLM identifier (e.g., "openai:gpt-realtime-2025-08-28")

    Returns:
        Tuple of (provider_order_index, model_name) for sorting
    """
    pk = get_provider_key(llm)
    idx = PROVIDER_ORDER_KEYS.index(pk) if pk in PROVIDER_ORDER_KEYS else 99
    return (idx, llm)


def add_model_columns(df: pd.DataFrame, llm_col: str = "llm") -> pd.DataFrame:
    """Add provider and model display columns to a DataFrame.

    Adds ``provider_key``, ``provider_display``, and ``model_display`` columns
    derived from the LLM identifier column.

    Args:
        df: DataFrame with an LLM identifier column
        llm_col: Name of the LLM column (default "llm")

    Returns:
        The DataFrame with new columns added (modified in place).
    """
    df["provider_key"] = df[llm_col].apply(get_provider_key)
    df["provider_display"] = df["provider_key"].apply(get_provider_display)
    df["model_display"] = df[llm_col].apply(lambda x: get_short_llm_name(x, max_len=25))
    return df


# =============================================================================
# LLM Color Configuration
# =============================================================================

# General color palette for LLMs (8 distinct colors)
COLOR_PALETTE = [
    "#4C72B0",  # Blue
    "#DD8452",  # Orange
    "#55A868",  # Green
    "#C44E52",  # Red
    "#8C564B",  # Brown
    "#8172B3",  # Purple
    "#CCB974",  # Gold
    "#64B5CD",  # Light blue
]

# Pre-defined colors for known LLMs (for consistency across all runs)
# Map provider prefixes and model names to specific colors
LLM_COLOR_MAP = {
    # OpenAI models - Blue
    "openai": "#4C72B0",
    # Gemini models - Green
    "gemini": "#55A868",
    # XAI/Grok models - Red
    "xai": "#C44E52",
    "grok": "#C44E52",
    # Anthropic models - Purple
    "anthropic": "#8172B3",
    "claude": "#8172B3",
    # Deepgram models - Orange
    "deepgram": "#DD8452",
    # AWS/Bedrock models - Brown
    "aws": "#8C564B",
    "bedrock": "#8C564B",
    # Azure models - Light blue
    "azure": "#64B5CD",
}


# =============================================================================
# LLM Color Functions
# =============================================================================


def get_llm_color(llm_name: str) -> str:
    """
    Get a deterministic color for an LLM based on its name.

    Colors are assigned based on:
    1. Known provider prefix (openai:, gemini:, xai:, etc.)
    2. Known model name patterns
    3. Hash of the LLM name for unknown models

    This ensures the same LLM always gets the same color regardless
    of the order in which LLMs are loaded.

    Args:
        llm_name: The LLM identifier (e.g., "openai:gpt-4", "gemini:gemini-live-2.5")

    Returns:
        Hex color string
    """
    if not llm_name:
        return COLOR_PALETTE[0]

    llm_lower = llm_name.lower()

    # Check for known provider prefix
    if ":" in llm_name:
        provider = llm_name.split(":")[0].lower()
        if provider in LLM_COLOR_MAP:
            return LLM_COLOR_MAP[provider]

    # Check for known patterns in the name
    for pattern, color in LLM_COLOR_MAP.items():
        if pattern in llm_lower:
            return color

    # Fall back to hash-based color assignment for unknown LLMs
    # Use hash of the name to ensure deterministic color
    color_index = hash(llm_name) % len(COLOR_PALETTE)
    return COLOR_PALETTE[color_index]


def _adjust_brightness(hex_color: str, factor: float) -> str:
    """Adjust color brightness. factor < 1 darkens, > 1 lightens."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    if factor > 1:
        t = min(factor - 1, 1.0)
        r = min(255, int(r + (255 - r) * t))
        g = min(255, int(g + (255 - g) * t))
        b = min(255, int(b + (255 - b) * t))
    else:
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))

    return f"#{r:02x}{g:02x}{b:02x}"


def get_llm_colors(llms: list) -> list:
    """
    Get colors for a list of LLMs with shade variation for same-provider models.

    When multiple models share a provider, they get evenly spaced darker-to-lighter
    variants of the provider's base color. Single models per provider use the
    base color unchanged.

    Args:
        llms: List of LLM identifiers

    Returns:
        List of hex color strings
    """
    from collections import defaultdict

    provider_groups: dict[str, list[str]] = defaultdict(list)
    for llm in llms:
        provider = llm.split(":")[0].lower() if ":" in llm else llm.lower()
        # Also check pattern-based matching for LLMs without provider prefix
        matched_provider = provider
        for pattern in LLM_COLOR_MAP:
            if pattern in llm.lower():
                matched_provider = pattern
                break
        provider_groups[matched_provider].append(llm)

    colors = {}
    for provider, group_llms in provider_groups.items():
        base_color = get_llm_color(group_llms[0])
        if len(group_llms) == 1:
            colors[group_llms[0]] = base_color
        else:
            for i, llm in enumerate(sorted(group_llms)):
                # Range from 0.75 (darker) to 1.25 (lighter)
                factor = 0.75 + (0.5 * i / (len(group_llms) - 1))
                colors[llm] = _adjust_brightness(base_color, factor)

    return [colors[llm] for llm in llms]


# =============================================================================
# Bar Style Configuration
# =============================================================================

# Default bar styling
BAR_STYLE = {
    "edgecolor": "white",
    "linewidth": 0.5,
}

# Complexity-specific styling
# - Control: lighter alpha (0.6), hatched (/)
# - Single ablations: medium alpha (0.7), various hatches
# - Pairwise ablations: medium-high alpha (0.8), various hatches
# - Regular: full alpha (0.9), solid (no hatch)
COMPLEXITY_STYLES = {
    "control": {"alpha": 0.6, "hatch": "/"},
    # Single-feature ablations
    "control_audio": {"alpha": 0.7, "hatch": "\\"},
    "control_accents": {"alpha": 0.7, "hatch": "."},
    "control_behavior": {"alpha": 0.7, "hatch": "+"},
    # Pairwise ablations
    "control_audio_accents": {"alpha": 0.8, "hatch": "x"},
    "control_audio_behavior": {"alpha": 0.8, "hatch": "o"},
    "control_accents_behavior": {"alpha": 0.8, "hatch": "*"},
    # Full
    "regular": {"alpha": 0.9, "hatch": ""},
}


def get_complexity_style(complexity: str) -> Dict:
    """Get the standard style (alpha, hatch) for a complexity level."""
    return COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})


def get_bar_style(complexity: str = None, **overrides) -> Dict:
    """
    Get complete bar styling for matplotlib bar plots.

    Args:
        complexity: Optional complexity level ("control" or "regular")
        **overrides: Additional style overrides (e.g., color=...)

    Returns:
        Dict with edgecolor, linewidth, alpha, hatch suitable for bar() kwargs
    """
    style = {**BAR_STYLE}
    if complexity:
        style.update(get_complexity_style(complexity))
    style.update(overrides)
    return style


def get_legend_patch(complexity: str, facecolor: str = "gray"):
    """
    Create a matplotlib Patch for legend with correct complexity styling.

    Args:
        complexity: "control" or "regular"
        facecolor: Color for the patch (default gray)

    Returns:
        matplotlib.patches.Patch object
    """
    from matplotlib.patches import Patch

    style = get_complexity_style(complexity)
    label = complexity.capitalize()
    return Patch(
        facecolor=facecolor,
        alpha=style["alpha"],
        hatch=style["hatch"],
        edgecolor=BAR_STYLE["edgecolor"],
        label=label,
    )


def style_axis(ax, grid: bool = True) -> None:
    """
    Apply consistent axis styling: hide top/right spines, optionally add grid.

    Args:
        ax: matplotlib Axes object
        grid: Whether to add horizontal grid lines (default True)
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)


SHORT_MODEL_NAMES: dict[str, str] = {
    "gemini-live-2.5-flash-native-audio": "gemini-live-2.5",
    "gpt-realtime-1.5": "gpt-realtime-1.5",
    "xai-realtime": "grok-voice",
}


def get_short_llm_name(llm_name: str, max_len: int = 20) -> str:
    """
    Get a shortened display name for an LLM.

    Uses curated ``SHORT_MODEL_NAMES`` when available, otherwise strips the
    provider prefix and truncates.

    Args:
        llm_name: Full LLM identifier (e.g., "openai:gpt-realtime-1.5")
        max_len: Maximum length for fallback truncation

    Returns:
        Shortened display name (e.g., "GPT Realtime 1.5")
    """
    if not llm_name:
        return "unknown"

    # Extract the model name after the provider prefix
    name = llm_name.split(":")[-1] if ":" in llm_name else llm_name

    if name in SHORT_MODEL_NAMES:
        return SHORT_MODEL_NAMES[name]

    # Fallback: truncate if needed
    if len(name) > max_len:
        return name[:max_len]
    return name


# =============================================================================
# Domain Task Counts
# =============================================================================


def get_domain_task_count(domain: str, split: Optional[str] = "base") -> int:
    """Get the number of tasks for a domain.

    Args:
        domain: The domain name (e.g., "airline", "retail", "telecom").
        split: The task split to use (default: "base"). If None, returns all tasks.

    Returns:
        The number of tasks in the domain/split.
    """
    from tau2.registry import registry

    tasks_loader = registry.get_tasks_loader(domain)
    tasks = tasks_loader(split)
    return len(tasks)


def get_domain_task_counts(
    domains: Optional[list] = None, split: Optional[str] = "base"
) -> Dict[str, int]:
    """Get task counts for multiple domains.

    Args:
        domains: List of domain names. If None, uses DOMAINS.
        split: The task split to use (default: "base"). If None, returns all tasks.

    Returns:
        Dictionary mapping domain name to task count.
    """
    if domains is None:
        domains = DOMAINS

    return {domain: get_domain_task_count(domain, split) for domain in domains}
