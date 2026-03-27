# Performance Analysis Guide

This guide explains how to add new analyses or improve existing ones in `performance_analysis.py`.

## Overview

The `performance_analysis.py` script follows a consistent pattern for each analysis:

```
Analysis
├── raw.csv          # Raw per-record data
├── analysis.csv     # Aggregated/pivoted statistics
└── *.pdf            # One or more visualization plots
```

All outputs for an analysis go into a **dedicated subdirectory** (e.g., `pass_k_by_domain/`, `action_success/`, `review_analysis/`).

---

## Architecture

### 1. Main Entry Point

The `analyze_results()` function orchestrates all analyses:

```python
def analyze_results(data_dir, output_dir, filter_domains):
    # 1. Load simulation results
    results = load_simulation_results(data_dir, filter_domains)
    
    # 2. Build core metrics DataFrame
    df_metrics, df_pass_hat_k = build_metrics_dataframe(results)
    
    # 3. Run each analysis in a try/except block
    try:
        analysis_dir = output_dir / "my_analysis"
        save_my_analysis_raw(analysis_dir, results)
        save_my_analysis(analysis_dir, df_raw)
        plot_my_analysis(analysis_dir, df_raw)
    except Exception as e:
        logger.error(f"Failed to generate my analysis: {e}")
```

### 2. Data Flow

```
Simulation Folders → load_simulation_results() → List[Tuple[params, Results]]
                                                          ↓
                                              build_metrics_dataframe()
                                                          ↓
                                              (df_metrics, df_pass_hat_k)
                                                          ↓
                                            Individual Analysis Functions
```

### 3. Core Data Structures

| Variable | Type | Description |
|----------|------|-------------|
| `results` | `List[Tuple[dict, Results]]` | List of (params, simulation_results) tuples |
| `params` | `dict` | Contains `llm`, `domain`, `speech_complexity`, `provider`, etc. |
| `Results` | `tau2.data_model.simulation.Results` | Contains list of `SimulationRun` objects |
| `SimulationRun` | Object | Has `ticks`, `reward_info`, `review`, `speech_env_params` |
| `df_metrics` | `pd.DataFrame` | Aggregated pass^k metrics per (domain, llm, complexity) |

---

## Adding a New Analysis

### Step 1: Create the Extract Function (Optional)

If your analysis needs to extract data from simulation runs:

```python
def extract_my_data(
    results: List[Tuple[dict, Results]]
) -> pd.DataFrame:
    """
    Extract raw data for my analysis.
    
    Returns a DataFrame with one row per [entity being analyzed].
    """
    rows = []
    for params, sim_results in results:
        for sim in sim_results.simulations:
            # Extract data from each simulation
            rows.append({
                "llm": params["llm"],
                "domain": params["domain"],
                "speech_complexity": params["speech_complexity"],
                "simulation_id": sim.simulation_id,
                # ... your specific fields ...
            })
    return pd.DataFrame(rows)
```

### Step 2: Create the Raw Data Saver

```python
def save_my_analysis_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """Save raw per-record data to {dir_name}_raw.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = extract_my_data(results)
    df_raw.to_csv(output_dir / f"{output_dir.name}_raw.csv", index=False)
    logger.info(f"Saved: {output_dir / f'{output_dir.name}_raw.csv'}")
    
    return df_raw
```

### Step 3: Create the Analysis Saver

```python
def save_my_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """Save aggregated analysis to {dir_name}_analysis.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate the raw data
    df_analysis = df_raw.groupby(["llm", "speech_complexity"]).agg(
        count=("some_field", "count"),
        mean_value=("some_field", "mean"),
        # ... your aggregations ...
    ).reset_index()
    
    df_analysis.to_csv(output_dir / f"{output_dir.name}_analysis.csv", index=False)
    logger.info(f"Saved: {output_dir / f'{output_dir.name}_analysis.csv'}")
```

### Step 4: Create the Plot Function

```python
def plot_my_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """Generate visualization for my analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if df_raw.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use unified styling (see Styling section below)
    llms = df_raw["llm"].unique()
    for llm in llms:
        color = get_llm_color(llm)
        for complexity in ["control", "regular"]:
            style = get_complexity_style(complexity)
            # ... plot your data ...
            ax.bar(
                x_pos,
                values,
                color=color,
                alpha=style["alpha"],
                hatch=style["hatch"],
                **BAR_STYLE,  # edgecolor="white", linewidth=0.5
            )
    
    # Apply standard axis styling
    style_axis(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / "my_plot.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'my_plot.pdf'}")
```

### Step 5: Register in analyze_results()

Add your analysis to the `analyze_results()` function:

```python
# ==========================================================================
# My New Analysis
# ==========================================================================
try:
    my_analysis_dir = output_dir / "my_analysis"
    df_raw = save_my_analysis_raw(my_analysis_dir, results)
    save_my_analysis(my_analysis_dir, df_raw)
    plot_my_analysis(my_analysis_dir, df_raw)
except Exception as e:
    logger.error(f"Failed to generate my analysis: {e}")
```

---

## Styling System

All plots should use the unified styling system for consistency.

### Configuration Constants

```python
# Bar styling defaults
BAR_STYLE = {
    "edgecolor": "white",
    "linewidth": 0.5,
}

# Complexity-specific styling
COMPLEXITY_STYLES = {
    "control": {"alpha": 0.6, "hatch": "/"},
    "regular": {"alpha": 0.9, "hatch": ""},
}
```

### Helper Functions

| Function | Purpose |
|----------|---------|
| `get_llm_color(llm_name)` | Get consistent color for an LLM |
| `get_complexity_style(complexity)` | Get alpha and hatch for control/regular |
| `get_bar_style(complexity)` | Get complete bar styling dict |
| `get_legend_patch(complexity, facecolor)` | Create legend patch with correct styling |
| `style_axis(ax)` | Hide top/right spines for clean look |

### Example Usage

```python
# Get LLM color (consistent across all plots)
color = get_llm_color("openai:gpt-4")

# Get complexity styling
style = get_complexity_style("control")  # {"alpha": 0.6, "hatch": "/"}

# Create a bar with full styling
ax.bar(
    x, y,
    color=color,
    alpha=style["alpha"],
    hatch=style["hatch"],
    edgecolor=BAR_STYLE["edgecolor"],
    linewidth=BAR_STYLE["linewidth"],
)

# Or use the convenience function
bar_style = get_bar_style("control")
ax.bar(x, y, color=color, **bar_style)

# Clean up axis
style_axis(ax)

# Create legend with correct complexity styling
legend_elements = [
    get_legend_patch("control"),
    get_legend_patch("regular"),
]
ax.legend(handles=legend_elements)
```

---

## Common Patterns

### 1. Grouped Bar Chart (Control vs Regular)

```python
llms = df["llm"].unique()
x = np.arange(len(llms))
bar_width = 0.35

for i, complexity in enumerate(["control", "regular"]):
    style = get_complexity_style(complexity)
    df_c = df[df["speech_complexity"] == complexity]
    values = [df_c[df_c["llm"] == llm]["value"].values[0] for llm in llms]
    colors = [get_llm_color(llm) for llm in llms]
    
    ax.bar(
        x + i * bar_width,
        values,
        bar_width,
        color=colors,
        alpha=style["alpha"],
        hatch=style["hatch"],
        **BAR_STYLE,
    )

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels([llm.split(":")[-1] for llm in llms], rotation=45, ha="right")
```

### 2. Horizontal Stacked Bar Chart

```python
categories = df["category"].unique()
y = np.arange(len(categories))

left = np.zeros(len(categories))
for value_type in value_types:
    widths = df[value_type].values
    ax.barh(y, widths, left=left, label=value_type, **BAR_STYLE)
    left += widths

ax.set_yticks(y)
ax.set_yticklabels(categories)
```

### 3. Multi-Panel Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, subset_name in zip(axes, ["All", "Critical", "Minor"]):
    # Plot each subset
    style_axis(ax)
    ax.set_title(subset_name)

plt.tight_layout()
plt.savefig(output_dir / "multi_panel.pdf", format="pdf", bbox_inches="tight")
```

---

## Data Access Patterns

### Accessing Simulation Data

```python
for params, sim_results in results:
    llm = params["llm"]
    domain = params["domain"]
    complexity = params["speech_complexity"]
    
    for sim in sim_results.simulations:
        sim_id = sim.simulation_id
        
        # Reward info
        if sim.reward_info:
            reward = sim.reward_info.reward
            action_checks = sim.reward_info.action_checks  # List[ActionCheck]
        
        # Review (LLM-based evaluation)
        if sim.review:
            errors = sim.review.errors  # List[ReviewError]
            summary = sim.review.summary
        
        # Speech environment parameters
        speech_params = getattr(sim, "speech_env_params", None)
        if speech_params:
            persona = speech_params.persona_name
            noise_file = speech_params.background_noise_file
        
        # Ticks (turn-by-turn data)
        for tick_idx, tick in enumerate(sim.ticks):
            agent_calls = tick.agent_tool_calls
            agent_results = tick.agent_tool_results
            user_calls = tick.user_tool_calls
            user_results = tick.user_tool_results
```

### Matching Tool Calls to Results

```python
# Agent tool calls and their results are parallel lists
for call, result in zip(tick.agent_tool_calls, tick.agent_tool_results):
    function_name = call.function
    arguments = call.arguments
    is_error = result.error
    error_message = result.content if result.error else None
```

---

## Checklist for New Analysis

- [ ] Create extraction function (if needed)
- [ ] Create `save_*_raw()` function → `raw.csv`
- [ ] Create `save_*_analysis()` function → `analysis.csv` (and other summary CSVs)
- [ ] Create `plot_*()` function(s) → `*.pdf`
- [ ] Use unified styling (`get_llm_color`, `get_complexity_style`, `BAR_STYLE`, `style_axis`)
- [ ] Add to `analyze_results()` in a try/except block
- [ ] Use a dedicated subdirectory for outputs
- [ ] Add logging with `logger.info(f"Saved: {path}")`
- [ ] Run `ruff check` to verify no lint errors

---

## Existing Analyses Reference

| Analysis | Directory | Outputs |
|----------|-----------|---------|
| Pass^k by Domain | `pass_k_by_domain/` | raw.csv, analysis.csv, pass_k.pdf, pass_1.pdf |
| Pass^k by Persona | `pass_k_by_persona/` | raw.csv, analysis.csv, pass_1.pdf |
| Pass^k by Background Noise | `pass_k_by_background_noise/` | raw.csv, analysis.csv, pass_1.pdf |
| Action Success (Tool Calls) | `action_success/` | raw.csv, analysis.csv, success_rates.csv, error_*.csv, action_success.pdf, success_summary.pdf |
| Reward Action Success | `reward_action_success/` | raw.csv, analysis.csv, success_rates.csv |
| Review Analysis | `review_analysis/` | raw_summary.csv, raw_errors.csv, error_*.csv, agent_*.pdf, user_*.pdf |
| Voice Metrics | (root) | voice_metrics*.csv, voice_metrics_*.pdf |
| Task Grid | (root) | task_pass_fail_grid.pdf |
| Termination Reasons | (root) | termination_reasons.csv, termination_reasons.pdf |
