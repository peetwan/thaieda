# ThaiEDA Visualization Modernization Research

**Date:** 2026-06-25
**Scope:** How to make ThaiEDA viz module produce more beautiful, modern, professional charts

---

## Current State (v1.0.1)

The `viz/__init__.py` module (1,306 lines) has 16 chart functions: `create_correlation_heatmap`, `create_boxplot`, `create_violinplot`, `create_distribution_histogram`, `create_missing_matrix`, `create_missing_heatmap`, `create_scatter_matrix`, `create_category_bar`, `create_timeseries_plot`, `create_decomposition_plot`, `create_acf_plot`, `create_wordcloud`, `create_length_histogram`, `create_top_tokens_chart`, `create_insight_chart`, `auto_visualize`, `auto_select_charts`.

Dark theme (`#1a1d23` bg, `#4dabf7` accent). Thai font auto-detection (Windows/Linux/macOS). All charts return base64 PNG for Jinja2 HTML report. matplotlib Agg backend.

**Strengths:** Comprehensive, offline, zero extra deps.  
**Weaknesses:** Static images, no interactivity, no hover/zoom, single-accent palette, small chart sizes.

---

## Research Findings

### 1. Modern Chart Library Comparison

| Library | Interactive | Thai Font | Output | Bundle Size | EDA Fit |
|---------|------------|-----------|--------|-------------|---------|
| **Plotly** | hover/zoom/pan | system fonts via browser | HTML div | ~3MB JS (CDN option) | Best |
| Seaborn | static | same as mpl | PNG | 0 | Good |
| Altair | limited | browser | HTML/JSON | ~1MB | Good |
| Bokeh | hover/zoom | browser | HTML | ~2MB | Good |

**ydata-profiling** (pandas-profiling successor): matplotlib static + optional Plotly interactive via `include_plotlyjs=cdn`.

**Recommendation: Add Plotly as optional backend (keep matplotlib default).**

Reasoning:
1. Plotly renders Thai text natively via browser fonts — **no font detection needed** for interactive charts.
2. `fig.to_html(full_html=False, include_plotlyjs=cdn)` returns a `<div>` for Jinja2.
3. Zoom/hover/pan makes EDA reports genuinely useful.
4. CDN option keeps report ~50KB vs ~3MB inline.
5. Plotly is industry standard for EDA dashboards (ydata-profiling, Datapane, Streamlit).

**Trade-off:** Interactive needs internet for CDN, or ~3MB offline (`include_plotlyjs=True`). Static PNG remains zero-dep fallback.

---

### 2. Chart Design Best Practices

**Color palettes:**
- Current: single accent `#4dabf7` + `#ff6b6b` (red outliers)
- Recommendation: 7-color colorblind-safe palette: `#4dabf7, #69db7c, #ffd43b, #ff6b6b, #da77f2, #ffa94d, #74c0fc` (Okabe-Ito inspired)
- Correlation heatmaps: keep `RdBu_r` (diverging) — industry standard

**Typography:**
- Current: `fontproperties=prop` on every text element — verbose
- Plotly: `font=dict(family="Sarabun, Noto Sans Thai, sans-serif")` set once on layout

**Layout/spacing:**
- Current: fixed figsize, `bbox_inches=tight`
- Plotly: `margin=dict(l=40, r=20, t=40, b=40)` + responsive `width: 100%`

---

### 3. Top 5 Concrete Improvements

#### Improvement 1: Add Plotly interactive backend (optional)
Add `create_*_interactive()` functions returning HTML divs. Gate behind `pip install thaieda[interactive]`.

#### Improvement 2: Better color palette
Replace single-accent with 7-color palette. Consistent colors across chart types.

#### Improvement 3: Add missing charts
- **Pair plot with hue** (group by target) — scatter matrix has no grouping currently
- **KDE/ridgeline plot** — distributions side-by-side
- **Sunburst/treemap** for hierarchical categorical
- **QQ plot** for normality

#### Improvement 4: Thai font for Plotly via CSS
Jinja2 template `<head>`: `@import` Google Fonts Sarabun. Set `layout.font.family`. Solves Thai rendering without font-path detection.

#### Improvement 5: Report mode selector
Add `report_format=static|interactive|both` to `DatasetReport.__init__`. Static=matplotlib PNGs, Interactive=Plotly divs, Both=interactive with PNG fallback.

---

### 4. Example Code: Improved Correlation Heatmap (Plotly)

```python
def create_correlation_heatmap_interactive(df, font_path=None):
    """Interactive correlation heatmap -- returns HTML div string."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Interactive charts require pip install thaieda[interactive]") from exc

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return ""

    corr = numeric.corr(numeric_only=True)
    labels = [str(c) for c in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).to_numpy(),
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="(%{y}, %{x}): %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        font=dict(family="Sarabun, Noto Sans Thai, sans-serif"),
        paper_bgcolor="#1a1d23", plot_bgcolor="#1a1d23",
        font_color="#e6e6e6",
        margin=dict(l=50, r=20, t=40, b=50),
        height=500,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="corr_heatmap")
```

---

### 5. Migration Path

**Phase 1 — Add Plotly backend (non-breaking):**
1. Add `plotly` to `pyproject.toml` `[interactive]` extra
2. Create `src/thaieda/viz/_interactive.py` — `*_interactive` functions return HTML divs
3. Add `setup_plotly_theme()` sets default layout template
4. Update `report/__init__.py`: add `report_format` param; when interactive call `_interactive` versions
5. Update Jinja2 template: `{{ chart_div|safe }}` blocks

**Phase 2 — Palette + new chart types:**
6. Define `_THAIEDA_PALETTE` (7 colors), update all matplotlib functions
7. Add `create_pairplot_hue()` — scatter matrix grouped by target
8. Add `create_kde_plot()` — ridge/KDE distribution comparison
9. Add `create_qq_plot()` — normality assessment

**Phase 3 — Polish:**
10. Add Google Fonts `@import` Sarabun in report template `<head>`
11. Add responsive CSS for chart containers
12. Document `pip install thaieda[interactive]` in README

**Changes in `viz/__init__.py`:**
- Keep all existing `create_*` functions (return base64 PNG) — no breaking changes
- Add `_interactive.py` sibling module with Plotly versions
- Add `_PALETTE` constant, update color references
- `auto_visualize`/`auto_select_charts` get `backend=static|interactive` param

**Changes in `report/__init__.py`:**
- `DatasetReport.__init__` gets `report_format=static` (default=current behavior)
- `_build_dataset_charts()` dispatches to matplotlib or Plotly
- `_render_html()` uses `|safe` filter for Plotly divs

---

### 6. Dependencies

```toml
[project.optional-dependencies]
interactive = ["plotly>=5.18.0"]
all = ["plotly>=5.18.0", "wordcloud>=1.9.0", "pythainlp>=4.0.0", ...]
```

- `plotly>=5.18.0` — only dep needed for interactive backend
- No changes to existing `pip install thaieda` (matplotlib path unchanged)
- Report size: ~50KB with CDN, ~3MB with `include_plotlyjs=True` (offline)

---

## Summary

| Aspect | Recommendation |
|--------|---------------|
| **Library** | Add Plotly optional interactive backend; keep matplotlib default |
| **Why Plotly** | Industry standard, native Thai via browser, Jinja2-compatible `to_html()` |
| **Top improvement** | Interactive correlation heatmap + boxplot with hover tooltips |
| **New charts** | Pair plot with hue, KDE/ridgeline, QQ plot |
| **Palette** | 7-color colorblind-safe palette |
| **Thai font** | CSS `@import` Sarabun + Plotly `font.family` |
| **Breaking changes** | None — all existing functions unchanged |
| **New dep** | `plotly>=5.18.0` (optional, `[interactive]` extra) |
