#!/usr/bin/env python3
"""Figures for the DLTlib cross-check subsection of report/hybrid_report.tex.

Reads analysis/dlt/overhead.csv (device,cells,t_step_s), fits the affine timing
model t = tau + cells/R per device, and writes two figures to report/img/:

  dlt_fit.png            per-device affine fit (t vs cells) + per-cell overhead view
  dlt_pred_vs_empirical  DLTlib+e0 predicted iGPU share vs the fraction-sweep peaks

The two-node DLT optimum used here is the closed form
    part_i = g_d/(g_i+g_d) + (g_d*e_d - g_i*e_i) / (L*(g_i+g_d)),   e = tau*R,
which reproduces DLTlib's SolveImage to 6 digits (validated against dlt_split).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "overhead.csv")
IMG = os.path.abspath(os.path.join(HERE, "..", "..", "report", "img"))

# Empirical fraction-sweep peaks (iGPU row share at the best throughput),
# from report/hybrid_report.tex Table 'tab:frac'.
EMPIRICAL = {8192: 0.06, 16384: 0.05, 24576: 0.05, 32768: 0.06, 40960: 0.05}
TARGET_NS = [8192, 16384, 24576, 32768, 40960]


def load():
    dev = {}
    with open(CSV) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, cells, t = line.split(",")
            dev.setdefault(name, []).append((float(cells), float(t)))
    return {k: np.array(v) for k, v in dev.items()}


def fit(pts):
    # Decoupled robust estimator (plain OLS of t on cells is leverage-biased by the
    # largest board): R = peak rate from the largest board; tau = mean residual
    # t - cells/R over the small boards (cells <= cells_max/256), where the fixed
    # overhead is a resolvable fraction of the step time.
    p = pts[np.argsort(pts[:, 0])]
    c, t = p[:, 0], p[:, 1]
    R = c[-1] / t[-1]
    small = c <= c[-1] / 256.0
    if not small.any():
        small = np.arange(len(c)) == 0
    tau = float(np.mean(t[small] - c[small] / R))
    return R, tau                     # R (cells/s), tau (s)


def igpu_share(L, Ri, ti, Rd, td):
    gi, gd = 1.0 / Ri, 1.0 / Rd
    ei, ed = max(ti, 0.0) * Ri, max(td, 0.0) * Rd
    naive = gd / (gi + gd)
    corr = (gd * ed - gi * ei) / (L * (gi + gd))
    return naive, naive + corr


def main():
    dev = load()
    R = {k: fit(v) for k, v in dev.items()}   # name -> (R, tau)
    Ri, ti = R["igpu"]
    Rd, td = R["dgpu"]
    os.makedirs(IMG, exist_ok=True)

    # ---- Figure 1: affine fit (t vs cells) + per-cell overhead view ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = {"igpu": "#1f77b4", "dgpu": "#d62728"}
    labels = {"igpu": "iGPU (UHD 630)", "dgpu": "dGPU (GTX 1660 Ti)"}
    xs = np.logspace(np.log10(2e5), np.log10(1.2e9), 200)
    for name in ("igpu", "dgpu"):
        c, t = dev[name][:, 0], dev[name][:, 1]
        Rn, tn = R[name]
        ax1.loglog(c, t * 1e3, "o", color=colors[name], label=labels[name])
        ax1.loglog(xs, (tn + xs / Rn) * 1e3, "-", color=colors[name], lw=1.2)
        # per-cell time (ns/cell): shows the fixed overhead as a small-N bump
        ax2.semilogx(c, t / c * 1e9, "o", color=colors[name])
        ax2.semilogx(xs, (tn + xs / Rn) / xs * 1e9, "-", color=colors[name],
                     label=f"{name}: R={Rn/1e9:.2f} Gc/s, "
                           + (rf"$\tau$={tn*1e6:.0f} $\mu$s" if tn > 0 else r"$\tau\approx0$"))
    ax1.set_xlabel("cells per step"); ax1.set_ylabel("time per step [ms]")
    ax1.set_title("(A) affine fit  $t = \\tau + $cells$/R$"); ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax2.set_xlabel("cells per step"); ax2.set_ylabel("time per cell [ns]")
    ax2.set_title("(B) per-cell time (overhead view)"); ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p1 = os.path.join(IMG, "dlt_fit.png")
    fig.savefig(p1, dpi=200); plt.close(fig)

    # ---- Figure 2: predicted vs empirical iGPU share ----
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    Ns = np.array(TARGET_NS)
    naive = np.array([igpu_share(n * n, Ri, ti, Rd, td)[0] for n in Ns]) * 100
    pred = np.array([igpu_share(n * n, Ri, ti, Rd, td)[1] for n in Ns]) * 100
    emp = np.array([EMPIRICAL[n] for n in Ns]) * 100
    ax.axhline(naive[0], ls=":", color="gray", label=f"naive $R_i/\\Sigma R$ = {naive[0]:.2f}%")
    ax.plot(Ns, pred, "o-", color="#1f77b4", label="DLTlib + fitted $e_0$")
    # The throughput-vs-fraction curve is flat-topped, so with best-of-5 boost
    # noise the winning grid point wanders ~1 step (the sweep sampled every 1%).
    ax.errorbar(Ns, emp, yerr=1.0, fmt="s", color="#2ca02c", capsize=3,
                label="empirical sweep peak ($\\pm$1 sweep step)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("grid size $N$ ($N\\times N$)")
    ax.set_ylabel("optimal iGPU row share [%]")
    ax.set_title("iGPU share: DLT prediction vs empirical optimum")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(IMG, "dlt_pred_vs_empirical.png")
    fig.savefig(p2, dpi=200); plt.close(fig)

    print("fit: iGPU R=%.3f Gc/s tau=%.0f us | dGPU R=%.3f Gc/s tau=%.0f us"
          % (Ri/1e9, ti*1e6, Rd/1e9, td*1e6))
    for n in TARGET_NS:
        nv, pr = igpu_share(n*n, Ri, ti, Rd, td)
        print("  N=%-6d naive=%.4f pred=%.4f empirical=%.2f"
              % (n, nv, pr, EMPIRICAL[n]))
    print("wrote", p1); print("wrote", p2)


if __name__ == "__main__":
    main()
