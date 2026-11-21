"""Charts built straight from SQL result frames (values are never typed by the LLM) and rendered with matplotlib."""
from __future__ import annotations

import io

import pandas as pd

KINDS = ("bar", "line", "pie")
MAX_POINTS = 40


def build_spec(df: pd.DataFrame, kind: str, title: str, x_label: str, y_label: str) -> dict:
    """First column = categories/x; every other numeric column = one series. Values come unchanged from df."""
    kind = (kind or "").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"chart_type must be one of {KINDS}")
    if df.empty or df.shape[1] < 2:
        raise ValueError("Query must return a label column plus at least one numeric column, with rows.")
    if len(df) > MAX_POINTS:
        raise ValueError(f"{len(df)} rows is too many to plot; aggregate further or LIMIT to {MAX_POINTS}.")
    series = []
    for col in df.columns[1:]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            series.append({"name": str(col), "values": [None if pd.isna(v) else float(v) for v in vals]})
    if not series:
        raise ValueError("No numeric column found after the first column.")
    if kind == "pie":
        series = series[:1]
        if any(v is not None and v < 0 for v in series[0]["values"]):
            raise ValueError("Pie charts need non-negative values.")
    return {"kind": kind, "title": title or "Chart", "x_label": x_label or str(df.columns[0]),
            "y_label": y_label or series[0]["name"], "labels": df.iloc[:, 0].astype(str).tolist(), "series": series}


def _fmt(v) -> str:
    return "" if v != v else f"{v:,.2f}".rstrip("0").rstrip(".")


def render_png(spec: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels, series, kind = spec["labels"], spec["series"], spec["kind"]
    nan = float("nan")
    vals = [[nan if v is None else v for v in s["values"]] for s in series]
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == "pie":
        ax.pie([0 if v != v else v for v in vals[0]], labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    elif kind == "bar":
        w = 0.8 / len(series)
        for i, (s, v) in enumerate(zip(series, vals)):
            bars = ax.bar([x + i * w for x in range(len(labels))], v, w, label=s["name"])
            if len(labels) <= 12 and len(series) == 1:
                ax.bar_label(bars, labels=[_fmt(x) for x in v], fontsize=8, padding=2)
        ax.set_xticks([x + w * (len(series) - 1) / 2 for x in range(len(labels))])
        ax.set_xticklabels(labels)
    else:
        for s, v in zip(series, vals):
            ax.plot(labels, v, marker="o", label=s["name"])
    if kind != "pie":
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}" if abs(x) >= 100 else f"{x:g}"))
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.12)
        if len(series) > 1:
            ax.legend()
        if len(labels) > 6 or max(len(l) for l in labels) > 10:
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title(spec["title"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()
