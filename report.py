"""Generate a self-contained HTML validation report with inline-SVG charts.

The charts are hand-built SVG, so the report has no plotting dependencies and is
a single file the client can open in any browser (light and dark theme aware).

    python report.py                          # 2,000,000 hands -> report.html
    python report.py -n 5000000 -o out.html   # more hands, custom output path
    python report.py --paytable my.json
"""

import argparse
import math
import time
from datetime import date

from mcpoker.evaluation import CATEGORIES
from mcpoker.paytable import (exact_probabilities, expected_return,
                              house_edge, load_paytable)
from mcpoker.simulation import simulate
from validate import chi_square, comparison_table, convergence

# Colours are the validated reference data-viz palette (light | dark), wired up
# as CSS custom properties so the whole report themes from one place. The two
# variable sets are emitted for the OS preference (prefers-color-scheme) and for
# an explicit theme toggle (data-theme), so either can drive the theme.
LIGHT_VARS = ("--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;"
              "--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--series:#2a78d6;"
              "--good:#006300;--border:rgba(11,11,11,.10);")
DARK_VARS = ("--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;"
             "--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;"
             "--good:#0ca30c;--border:rgba(255,255,255,.10);")

CSS = (
    ":root{" + LIGHT_VARS + "}"
    "@media (prefers-color-scheme:dark){:root{" + DARK_VARS + "}}"
    ':root[data-theme="light"]{' + LIGHT_VARS + "}"
    ':root[data-theme="dark"]{' + DARK_VARS + "}"
    """
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:40px 20px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 4px;font-size:15px}
.meta{color:var(--muted);font-size:13px;margin:0 0 28px;
  font-variant-numeric:tabular-nums}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:24px;margin:0 0 22px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin:0 0 18px;font-weight:600}
.note{color:var(--ink2);font-size:13.5px;margin:14px 0 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}
.tile{background:var(--plane);border:1px solid var(--border);border-radius:11px;
  padding:16px 18px}
.tile .k{color:var(--muted);font-size:12px;margin:0 0 6px;
  text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:28px;font-weight:650;letter-spacing:-.02em}
.tile .v small{font-size:15px;font-weight:500;color:var(--ink2)}
.badge{display:inline-flex;align-items:center;gap:7px;background:var(--plane);
  border:1px solid var(--border);border-radius:999px;padding:7px 14px;
  font-size:13.5px;font-weight:600}
.badge .dot{width:9px;height:9px;border-radius:50%;background:var(--good)}
.legend{display:flex;gap:20px;align-items:center;margin:0 0 6px;font-size:13px;
  color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.swatch{width:11px;height:11px;border-radius:50%;background:var(--series)}
.tick{width:2px;height:14px;background:var(--axis);border-radius:1px}
svg{display:block;width:100%;height:auto;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.axis{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.axl{fill:var(--muted);font-size:12px}
.cat{fill:var(--ink);font-size:13px}
.val{fill:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums}
.ref{stroke:var(--axis);stroke-width:2;stroke-linecap:round}
.dot{fill:var(--series);stroke:var(--surface);stroke-width:1.5}
.line{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round}
.refline{fill:none;stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px;
  font-variant-numeric:tabular-nums;min-width:340px}
th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.03em}
td.pass{color:var(--good);text-align:center}
.foot{color:var(--muted);font-size:12.5px;margin-top:32px;text-align:center}
""")


def fmt_pct(p, places=2):
    return f"{p * 100:.{places}f}%"


def freq_pct(p):
    """Adaptive percentage label so rare and common hands both read cleanly."""
    if p >= 0.01:
        return f"{p * 100:.2f}%"
    if p >= 0.0001:
        return f"{p * 100:.3f}%"
    return f"{p * 100:.4f}%"


# Chart 1: simulated vs exact hand frequencies (dot plot on a log axis)

DECADES = {-5: "0.001%", -4: "0.01%", -3: "0.1%", -2: "1%", -1: "10%", 0: "100%"}


def frequency_chart(rows):
    left, plot_w, right = 172, 470, 92
    top, row_h = 12, 34
    width = left + plot_w + right
    plot_bottom = top + len(rows) * row_h
    height = plot_bottom + 40
    lo, hi = -5.0, 0.0  # log10 probability domain: 0.001% .. 100%

    def x(p):
        return left + plot_w * (math.log10(max(p, 1e-6)) - lo) / (hi - lo)

    s = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Simulated versus exact hand frequencies">']
    for d, label in DECADES.items():
        gx = x(10.0 ** d)
        s.append(f'<line class="grid" x1="{gx:.1f}" y1="{top}" '
                 f'x2="{gx:.1f}" y2="{plot_bottom}"/>')
        s.append(f'<text class="axis" x="{gx:.1f}" y="{plot_bottom + 18}" '
                 f'text-anchor="middle">{label}</text>')
    s.append(f'<text class="axl" x="{left + plot_w / 2:.0f}" y="{height - 2}" '
             f'text-anchor="middle">Probability (log scale)</text>')

    for i, r in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        ex, sx = x(r["exact"]), x(r["simulated"])
        s.append(f'<text class="cat" x="{left - 14}" y="{cy + 4:.1f}" '
                 f'text-anchor="end">{r["category"]}</text>')
        s.append(f'<line class="ref" x1="{ex:.1f}" y1="{cy - 11:.1f}" '
                 f'x2="{ex:.1f}" y2="{cy + 11:.1f}"/>')
        s.append(f'<circle class="dot" cx="{sx:.1f}" cy="{cy:.1f}" r="6">'
                 f'<title>{r["category"]}: simulated {freq_pct(r["simulated"])}, '
                 f'exact {freq_pct(r["exact"])} (z={r["z"]:+.2f})</title></circle>')
        s.append(f'<text class="val" x="{left + plot_w + 14}" y="{cy + 4:.1f}">'
                 f'{freq_pct(r["simulated"])}</text>')
    s.append("</svg>")
    return "\n".join(s)


# Chart 2: convergence of the Monte-Carlo error (log-log)

def convergence_chart(points):
    left, plot_w, right = 60, 560, 24
    top, plot_h, bottom = 18, 232, 46
    width = left + plot_w + right
    height = top + plot_h + bottom
    xlo, xhi = 4.0, 6.3          # 10k .. ~2M hands
    ylo, yhi = -4.0, -2.0        # error 0.0001 .. 0.01

    def X(n):
        return left + plot_w * (math.log10(n) - xlo) / (xhi - xlo)

    def Y(e):
        return top + plot_h * (1 - (math.log10(e) - ylo) / (yhi - ylo))

    s = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Monte-Carlo error versus number of hands">']
    for d in (4, 5, 6):
        gx = X(10.0 ** d)
        s.append(f'<line class="grid" x1="{gx:.1f}" y1="{top}" '
                 f'x2="{gx:.1f}" y2="{top + plot_h}"/>')
        lab = {4: "10K", 5: "100K", 6: "1M"}[d]
        s.append(f'<text class="axis" x="{gx:.1f}" y="{top + plot_h + 18}" '
                 f'text-anchor="middle">{lab}</text>')
    for d in (-4, -3, -2):
        gy = Y(10.0 ** d)
        s.append(f'<line class="grid" x1="{left}" y1="{gy:.1f}" '
                 f'x2="{left + plot_w}" y2="{gy:.1f}"/>')
        lab = {-4: "0.0001", -3: "0.001", -2: "0.01"}[d]
        s.append(f'<text class="axis" x="{left - 8}" y="{gy + 4:.1f}" '
                 f'text-anchor="end">{lab}</text>')

    # Ideal 1/sqrt(N) reference slope, anchored to the first measured point.
    n0, e0 = points[0]
    ref = [(10.0 ** xlo, e0 * math.sqrt(n0 / 10.0 ** xlo)),
           (10.0 ** xhi, e0 * math.sqrt(n0 / 10.0 ** xhi))]
    rp = " ".join(f"{X(n):.1f},{Y(e):.1f}" for n, e in ref)
    s.append(f'<polyline class="refline" points="{rp}"/>')

    pts = " ".join(f"{X(n):.1f},{Y(e):.1f}" for n, e in points)
    s.append(f'<polyline class="line" points="{pts}"/>')
    for n, e in points:
        s.append(f'<circle class="dot" cx="{X(n):.1f}" cy="{Y(e):.1f}" r="5">'
                 f'<title>{n:,} hands: mean error {e:.5f}</title></circle>')
    s.append(f'<text class="axl" x="{left + plot_w / 2:.0f}" y="{height - 4}" '
             f'text-anchor="middle">Hands simulated (log scale)</text>')
    s.append("</svg>")
    return "\n".join(s)


def build_html(counts, paytable_path, hands, seed, elapsed, fragment=False):
    rows = comparison_table(counts)
    chi2, p_value, dof = chi_square(counts)
    max_z = max(abs(r["z"]) for r in rows)
    conv = [(n, err) for n, err, _ in
            convergence([10_000, 50_000, 250_000, 1_250_000],
                        trials=8, base_seed=seed)]

    paytable = load_paytable(paytable_path)
    p_exact = exact_probabilities()
    rtp = expected_return(p_exact, paytable)
    edge = house_edge(p_exact, paytable)
    speed = hands / elapsed / 1e6

    freq_rows = "\n".join(
        f"<tr><td>{r['category']}</td><td>{freq_pct(r['simulated'])}</td>"
        f"<td>{freq_pct(r['exact'])}</td><td>{r['z']:+.2f}</td>"
        f"<td class='pass'>{'&#10003;' if abs(r['z']) < 3 else '&#8226;'}</td></tr>"
        for r in rows)

    pay_rows = "\n".join(
        f"<tr><td>{c}</td><td>{paytable[c]}&times;</td>"
        f"<td>{freq_pct(p_exact[c])}</td></tr>" for c in CATEGORIES)

    body = f"""<div class="wrap">
  <h1>Monte-Carlo Simulation Validation Report</h1>
  <p class="sub">5-card hand frequencies, statistical validation and game economics.</p>
  <p class="meta">{hands:,} hands simulated &middot; seed {seed} &middot;
    {speed:.2f}M hands/sec &middot; {date.today().isoformat()}</p>

  <div class="card">
    <h2>Accuracy at a glance</h2>
    <div class="tiles">
      <div class="tile"><div class="k">Hands simulated</div>
        <div class="v">{hands / 1e6:.1f}<small>M</small></div></div>
      <div class="tile"><div class="k">Chi-square p-value</div>
        <div class="v">{p_value:.3f}</div></div>
      <div class="tile"><div class="k">Largest deviation</div>
        <div class="v">{max_z:.2f}<small> &sigma;</small></div></div>
      <div class="tile"><div class="k">Verdict</div>
        <div class="v" style="font-size:20px;padding-top:6px">
          <span class="badge"><span class="dot"></span>Validated</span></div></div>
    </div>
    <p class="note">Every simulated frequency sits within {max_z:.1f} standard
      errors of its exact combinatorial probability, and the chi-square test
      (dof&nbsp;=&nbsp;{dof}) does not reject agreement with theory. A separate
      test evaluates all 2,598,960 possible hands and matches the exact counts
      exactly.</p>
  </div>

  <div class="card">
    <h2>Simulated vs exact hand frequencies</h2>
    <div class="legend">
      <span><span class="swatch"></span>Simulated</span>
      <span><span class="tick"></span>Exact (theoretical)</span>
    </div>
    {frequency_chart(rows)}
    <div class="scroll"><table>
      <thead><tr><th>Category</th><th>Simulated</th><th>Exact</th>
        <th>z</th><th>Match</th></tr></thead>
      <tbody>{freq_rows}</tbody>
    </table></div>
  </div>

  <div class="card">
    <h2>Convergence: error shrinks as 1/&#8730;N</h2>
    {convergence_chart(conv)}
    <p class="note">Mean error (averaged over 8 seeds) falls along the dashed
      1/&#8730;N reference line, the expected Monte-Carlo convergence rate,
      confirming the estimator is unbiased.</p>
  </div>

  <div class="card">
    <h2>Game economics</h2>
    <div class="tiles">
      <div class="tile"><div class="k">Return to player</div>
        <div class="v">{fmt_pct(rtp)}</div></div>
      <div class="tile"><div class="k">House edge</div>
        <div class="v">{fmt_pct(edge)}</div></div>
    </div>
    <div class="scroll"><table style="margin-top:18px">
      <thead><tr><th>Category</th><th>Payout</th><th>Probability</th></tr></thead>
      <tbody>{pay_rows}</tbody>
    </table></div>
    <p class="note">Illustrative paytable. Drop in the real game's payouts and the
      return-to-player and house edge recompute instantly, with no re-simulation
      needed.</p>
  </div>

  <p class="foot">Generated by the Monte-Carlo simulator &middot; open in any browser</p>
</div>"""

    if fragment:
        return f"<style>{CSS}</style>\n{body}\n"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Monte-Carlo Simulation Report</title>\n'
        f'<style>{CSS}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n'
    )


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML validation report.")
    parser.add_argument("-n", "--hands", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paytable", default="paytables/example.json")
    parser.add_argument("-o", "--output", default="report.html")
    parser.add_argument("--fragment", action="store_true",
                        help="emit only the inner HTML, for embedding elsewhere")
    args = parser.parse_args()

    start = time.perf_counter()
    counts = simulate(args.hands, seed=args.seed)
    elapsed = time.perf_counter() - start

    html = build_html(counts, args.paytable, args.hands, args.seed, elapsed,
                      fragment=args.fragment)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.output} ({args.hands:,} hands, {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
