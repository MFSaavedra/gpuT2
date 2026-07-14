#!/usr/bin/env python3
"""Figure for the iGPU+dGPU fraction-sweep section of report/hybrid_report.tex.

Reads results/linux/frac_sweep.csv (written by scripts/sweep_fracs.sh: the binary's
12-column --csv schema plus a leading igpu_frac column, REPS rows per point) and
writes report/img/hybrid_frac_sweep.png:

  (A) steady-state throughput vs iGPU row share, one curve per grid size N —
      best-of-reps per point, an ordinal light->dark ramp so the curve order
      encodes N, and the naive-DLT share R_i/(R_i+R_d) as a reference line.
  (B) best static split vs pure dGPU as a share of the DLT ceiling R_i+R_d, on a
      single axis. (An earlier revision twin-axed "gain over dGPU" against
      "share of ceiling"; those are the same curve up to the constant dGPU/ceiling
      factor, so the two lines rendered on top of each other — the gain is the
      *gap* between the two curves here, annotated at the ends.)

Also prints the Table 'tab:frac' summary rows (and the peak shares that feed
EMPIRICAL in analysis/dlt/plot_dlt.py), so report numbers can be synced.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "linux", "frac_sweep.csv")
OUT = os.path.abspath(os.path.join(HERE, "..", "report", "img", "hybrid_frac_sweep.png"))

R_IGPU = 2.1e3  # Mcells/s; the iGPU halo rate from the DLT fit (report Eq. naive2gpu)

# Ordinal blue ramp (one hue, light->dark encodes N) + chart chrome.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK2, MUTED, GRIDC, AXISC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load():
    """-> {N: {frac: best-of-reps Mcells/s (kernel)}}"""
    best = defaultdict(dict)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            n, frac = int(row["rows"]), float(row["igpu_frac"])
            mc = float(row["mcells_kernel"])
            if mc > best[n].get(frac, 0.0):
                best[n][frac] = mc
    return dict(sorted(best.items()))


def style(ax):
    ax.grid(True, color=GRIDC, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(AXISC)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    best = load()
    ns = list(best)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=200)
    fig.patch.set_facecolor("white")

    # ---- (A) throughput vs split, per grid size ----
    for i, n in enumerate(ns):
        pts = sorted(best[n].items())
        fr = [f * 100 for f, _ in pts]
        gc = [m / 1e3 for _, m in pts]
        axa.plot(fr, gc, "-o", color=RAMP[i], lw=2, ms=5,
                 markeredgecolor="white", markeredgewidth=1.1,
                 solid_capstyle="round", label=f"$N$={n}")
    # Reference the naive share against the mean pure-dGPU rate over the grids.
    r_d = sum(best[n][0.0] for n in ns) / len(ns)
    naive = 100 * R_IGPU / (R_IGPU + r_d)
    axa.axvline(naive, color=MUTED, lw=1, ls=(0, (4, 3)))
    axa.text(naive + 0.15, axa.get_ylim()[0] + 0.5,
             f"naive DLT\n$R_i/\\Sigma R$ = {naive:.1f}%",
             color=INK2, fontsize=7.5, va="bottom")
    axa.set_xlabel("iGPU row share (%)", color=INK2, fontsize=9)
    axa.set_ylabel("throughput (Gcells/s, kernel)", color=INK2, fontsize=9)
    axa.set_title("(A) Throughput vs split, per grid size", color=INK, fontsize=10)
    axa.legend(fontsize=7.5, loc="lower left", edgecolor=AXISC,
               labelcolor=INK2, framealpha=1.0)
    style(axa)

    # ---- (B) share of the DLT ceiling, single axis ----
    dgpu = [best[n][0.0] for n in ns]
    peak = [max(best[n].values()) for n in ns]
    ceil = [d + R_IGPU for d in dgpu]
    hyb_pct = [100 * p / c for p, c in zip(peak, ceil)]
    dgpu_pct = [100 * d / c for d, c in zip(dgpu, ceil)]
    win = [100 * (p / d - 1) for p, d in zip(peak, dgpu)]

    axb.axhline(100, color=AXISC, lw=1)
    axb.text(ns[0], 100.15, "DLT ceiling  $R_i+R_d$", color=INK2, fontsize=7.5,
             va="bottom")
    axb.plot(ns, hyb_pct, "-o", color=RAMP[2], lw=2, ms=5.5,
             markeredgecolor="white", markeredgewidth=1.1, label="best static split")
    axb.plot(ns, dgpu_pct, "-o", color=MUTED, lw=2, ms=5,
             markeredgecolor="white", markeredgewidth=1.1, label="pure dGPU")
    # The gain over pure dGPU is the gap between the curves; annotate the ends.
    for i in sorted({0, win.index(max(win))}):
        axb.annotate(f"+{win[i]:.1f}%", (ns[i], hyb_pct[i]), (0, 8),
                     textcoords="offset points", ha="center",
                     color=INK2, fontsize=8)
    axb.set_xticks(ns)
    axb.set_xticklabels([str(n) for n in ns])
    axb.set_xlabel("grid size $N$  ($N \\times N$)", color=INK2, fontsize=9)
    axb.set_ylabel("% of DLT ceiling  $R_i+R_d$", color=INK2, fontsize=9)
    axb.set_title("(B) Best split vs pure dGPU, share of the ceiling",
                  color=INK, fontsize=10)
    axb.set_ylim(min(dgpu_pct) - 1.2, 101.4)
    axb.legend(fontsize=7.5, loc="center right", edgecolor=AXISC,
               labelcolor=INK2, framealpha=1.0)
    style(axb)

    fig.tight_layout()
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)

    # tab:frac / plot_dlt.py EMPIRICAL sync
    print(f"\n{'N':>6} {'dGPU':>6} {'best':>6} {'share':>6} {'win':>6} "
          f"{'ceiling':>7} {'%ceil':>6}")
    for i, n in enumerate(ns):
        share = max(best[n], key=best[n].get)
        print(f"{n:>6} {dgpu[i]/1e3:>6.2f} {peak[i]/1e3:>6.2f} {share*100:>5.0f}% "
              f"{win[i]:>+5.1f}% {ceil[i]/1e3:>7.2f} {hyb_pct[i]:>5.1f}%")


if __name__ == "__main__":
    main()
