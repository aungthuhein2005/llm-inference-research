"""
Draw the README charts from results/benchmark.csv.

Writes results/throughput.svg and results/latency.svg. Standard library only;
the SVGs switch to dark colors when the viewer's system is in dark mode.

    python make_charts.py
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

WIDTH, HEIGHT = 720, 380
LEFT, RIGHT, TOP, BOTTOM = 64, 176, 76, 56
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM

STYLE = """
<style>
  .surface { fill: #fcfcfb; }
  .ink-1 { fill: #0b0b0b; }
  .ink-2 { fill: #52514e; }
  .muted { fill: #898781; }
  .grid { stroke: #e1e0d9; }
  .axis { stroke: #c3c2b7; }
  .ref { stroke: #898781; }
  .s1 { stroke: #2a78d6; }
  .s1-fill { fill: #2a78d6; }
  .s2 { stroke: #eb6834; }
  .s2-fill { fill: #eb6834; }
  .ring { stroke: #fcfcfb; }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .tick { font-size: 12px; font-variant-numeric: tabular-nums; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink-1 { fill: #f0efec; }
    .ink-2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .s1 { stroke: #3987e5; }
    .s1-fill { fill: #3987e5; }
    .s2 { stroke: #d95926; }
    .s2-fill { fill: #d95926; }
    .ring { stroke: #1a1a19; }
  }
</style>
"""


def load_rows():
    with open(RESULTS_DIR / "benchmark.csv", newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def x_pos(index, count):
    # Concurrency levels double each step, so they sit evenly spaced (a log2 axis)
    return LEFT + index * PLOT_W / (count - 1)


def y_pos(value, y_max):
    return TOP + PLOT_H - value / y_max * PLOT_H


def frame(title, subtitle, y_max, y_step, y_format, concurrencies, legend):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{title}">',
        STYLE,
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
        f'<text class="ink-1" x="{LEFT}" y="28" font-size="16" font-weight="600">{title}</text>',
        f'<text class="ink-2" x="{LEFT}" y="48" font-size="13">{subtitle}</text>',
    ]

    # Legend in one row under the subtitle
    legend_x = LEFT
    for css_class, label, dashed in legend:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<line class="{css_class}" x1="{legend_x}" y1="64" x2="{legend_x + 20}" y2="64" stroke-width="2"{dash}/>'
        )
        parts.append(f'<text class="ink-2" x="{legend_x + 26}" y="68" font-size="12">{label}</text>')
        legend_x += 26 + len(label) * 7 + 24

    # Horizontal gridlines and y tick labels
    value = 0
    while value <= y_max:
        y = y_pos(value, y_max)
        line_class = "axis" if value == 0 else "grid"
        parts.append(f'<line class="{line_class}" x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" stroke-width="1"/>')
        parts.append(f'<text class="muted tick" x="{LEFT - 10}" y="{y + 4:.1f}" text-anchor="end">{y_format(value)}</text>')
        value += y_step

    # x tick labels and axis title
    for i, concurrency in enumerate(concurrencies):
        parts.append(
            f'<text class="muted tick" x="{x_pos(i, len(concurrencies)):.1f}" y="{TOP + PLOT_H + 20}" text-anchor="middle">{concurrency}</text>'
        )
    parts.append(
        f'<text class="ink-2" x="{LEFT + PLOT_W / 2}" y="{HEIGHT - 12}" font-size="12" text-anchor="middle">Concurrent requests</text>'
    )
    return parts


def series(values, y_max, stroke_class, fill_class, dashed=False, markers=True):
    count = len(values)
    points = " ".join(f"{x_pos(i, count):.1f},{y_pos(v, y_max):.1f}" for i, v in enumerate(values))
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    parts = [
        f'<polyline class="{stroke_class}" points="{points}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
    ]
    if markers:
        for i, v in enumerate(values):
            # 8px marker with a 2px surface ring so it separates from the line
            parts.append(
                f'<circle class="{fill_class} ring" cx="{x_pos(i, count):.1f}" cy="{y_pos(v, y_max):.1f}" r="5" stroke-width="2"/>'
            )
    return parts


def end_label(value, y_max, count, lines, css_class="ink-1"):
    x = x_pos(count - 1, count) + 14
    y = y_pos(value, y_max)
    parts = [f'<text class="{css_class}" x="{x:.1f}" y="{y - 2:.1f}" font-size="13" font-weight="600">{lines[0]}</text>']
    if len(lines) > 1:
        parts.append(f'<text class="ink-2" x="{x:.1f}" y="{y + 14:.1f}" font-size="12">{lines[1]}</text>')
    return parts


def throughput_chart(rows):
    concurrencies = [int(r["concurrency"]) for r in rows]
    measured = [r["output_tokens_per_second"] for r in rows]
    base = measured[0]
    ideal = [base * c / concurrencies[0] for c in concurrencies]
    y_max = 2500
    count = len(rows)

    parts = frame(
        "Output throughput vs. concurrency",
        "Total output tokens per second across all in-flight requests",
        y_max,
        500,
        lambda v: f"{v:,.0f}",
        concurrencies,
        [("s1", "Measured", False), ("ref", "Ideal linear scaling", True)],
    )
    parts += series(ideal, y_max, "ref", None, dashed=True, markers=False)
    parts += series(measured, y_max, "s1", "s1-fill")
    parts += end_label(ideal[-1], y_max, count, [f"Ideal {ideal[-1]:,.0f} tok/s", f"{concurrencies[-1] // concurrencies[0]}× concurrency 1"], "ink-2")
    parts += end_label(measured[-1], y_max, count, [f"{measured[-1]:,.0f} tok/s", f"{measured[-1] / base:.1f}× concurrency 1"])
    # Label the starting point too, so the baseline value is readable without the table
    parts.append(
        f'<text class="ink-1" x="{x_pos(0, count) + 10:.1f}" y="{y_pos(measured[0], y_max) - 12:.1f}" font-size="12">{measured[0]:,.0f} tok/s</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def latency_chart(rows):
    concurrencies = [int(r["concurrency"]) for r in rows]
    p50 = [r["latency_p50_s"] * 1000 for r in rows]
    p95 = [r["latency_p95_s"] * 1000 for r in rows]
    y_max = 800
    count = len(rows)

    parts = frame(
        "Request latency vs. concurrency",
        "End-to-end latency per request, 64 output tokens",
        y_max,
        200,
        lambda v: f"{v:.0f} ms",
        concurrencies,
        [("s1", "p50", False), ("s2", "p95", False)],
    )
    parts += series(p50, y_max, "s1", "s1-fill")
    parts += series(p95, y_max, "s2", "s2-fill")
    parts += end_label(p95[-1], y_max, count, [f"p95 {p95[-1]:.0f} ms", f"from {p95[0]:.0f} ms at 1"])
    parts += end_label(p50[-1], y_max, count, [f"p50 {p50[-1]:.0f} ms", f"from {p50[0]:.0f} ms at 1"])
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    rows = load_rows()
    (RESULTS_DIR / "throughput.svg").write_text(throughput_chart(rows), encoding="utf-8")
    (RESULTS_DIR / "latency.svg").write_text(latency_chart(rows), encoding="utf-8")
    print("Charts saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
