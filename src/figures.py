"""Generates the five figures of the ARGUS + GRAIN paper (300 dpi)."""

from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
CW = 3.4
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2, "axes.linewidth": 0.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.3,
    "axes.labelsize": 7.4, "axes.titlesize": 7.6, "savefig.dpi": 300,
    "figure.dpi": 300, "lines.linewidth": 1.0, "legend.frameon": False,
})
GREY = "#4d4d4d"
MCOL = {"STOREALL": "#d9d9d9", "DECIMATE": "#bdbdbd", "SENDDELTA": "#8c8c8c",
        "RF": "#4c72b0", "ARGUS": "#c0392b"}
MNAME = {"STOREALL": "store-all", "DECIMATE": "decimate", 
         "SENDDELTA": "send-on-delta", "RF": "random forest", "ARGUS": "ARGUS"}
ECOL = {"RAWSCAN": "#bdbdbd", "TIMEIDX": "#8c8c8c", "MATVIEW": "#4c72b0",
        "GRAIN": "#c0392b"}
ENAME = {"RAWSCAN": "raw scan", "TIMEIDX": "time index",
         "MATVIEW": "mat. view", "GRAIN": "GRAIN"}
HATCH = {"STOREALL": "", "DECIMATE": "//", "SENDDELTA": "..", "RF": "\\\\",
         "ARGUS": "xx", "RAWSCAN": "", "TIMEIDX": "//", "MATVIEW": "..",
         "GRAIN": "xx"}


def _s():
    return json.load(open(f"{RES}/summary.json"))


# ----------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(CW, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.4); ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=6.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=0.6, edgecolor=GREY, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arr(x1, y1, x2, y2, ls="-", c=GREY, lw=0.7):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=lw,
                                     linestyle=ls, color=c))

    box(0.15, 9.15, 9.7, 2.1, "", "#f2f2f2")
    ax.text(0.35, 11.05, "LAYER 1  Field sensing layer", fontsize=6.6,
            fontweight="bold", va="top")
    for k, lab in enumerate(["plot 1\n8 depths", "plot 2\n8 depths",
                             "plot 3\n8 depths", "plots 4-5\n8 depths"]):
        box(0.45 + 2.36 * k, 9.35, 2.05, 1.2, lab, "white", fs=5.6)
    ax.text(0.45, 8.58, "water content, ion content, soil temperature (30 min)",
            fontsize=5.2, va="center", style="italic")

    for x in (2.0, 5.0, 8.0):
        arr(x, 9.15, x, 8.15)

    box(0.15, 5.95, 9.7, 2.25, "", "#e8e8e8")
    ax.text(0.35, 8.02, "LAYER 2  ARGUS three-way identification",
            fontsize=6.6, fontweight="bold", va="top")
    box(0.45, 6.15, 2.9, 1.35, "dual predictor\ntemporal + peer\n(2)", "white", fs=5.4)
    box(3.55, 6.15, 2.9, 1.35, "decision-aware\ntolerance  (3)", "white", fs=5.4)
    box(6.65, 6.15, 2.9, 1.35, "urgency gate\n(1)", "white", fs=5.4)
    arr(3.35, 6.82, 3.55, 6.82); arr(6.45, 6.82, 6.65, 6.82)

    arr(1.7, 5.95, 1.7, 5.0); ax.text(1.82, 5.42, "urgent", fontsize=5.4)
    arr(5.0, 5.95, 5.0, 5.0); ax.text(5.12, 5.42, "strategic", fontsize=5.4)
    ax.plot([7.9, 8.3], [5.85, 5.45], color=GREY, lw=0.7)
    ax.plot([7.9, 8.3], [5.45, 5.85], color=GREY, lw=0.7)
    ax.text(8.45, 5.65, "redundant,\ndiscarded", fontsize=5.2, va="center")

    box(0.15, 2.85, 9.7, 2.1, "", "#dedede")
    ax.text(0.35, 4.75, "LAYER 3  GRAIN retrieval engine", fontsize=6.6,
            fontweight="bold", va="top")
    box(0.45, 3.05, 4.4, 1.3, "inverted event index\n(node, type) -> times", "white", fs=5.4)
    box(5.05, 3.05, 4.5, 1.3, "per-plot daily sketches\ncount, sum, min/max, hist", "white", fs=5.4)
    arr(4.85, 3.70, 5.05, 3.70)

    arr(2.6, 2.85, 2.6, 2.2); arr(7.3, 2.85, 7.3, 2.2)
    box(0.15, 0.3, 4.6, 1.9, "", "#f2f2f2")
    ax.text(0.35, 2.05, "Urgent insight", fontsize=6.3, fontweight="bold", va="top")
    box(0.45, 0.5, 4.0, 1.15, "Q1 latest event\nQ2 events in window", "white", fs=5.4)
    box(5.25, 0.3, 4.6, 1.9, "", "#f2f2f2")
    ax.text(5.45, 2.05, "Strategic insight", fontsize=6.3, fontweight="bold", va="top")
    box(5.55, 0.5, 4.0, 1.15, "Q3 window mean, p90\nQ4 degree-hours", "white", fs=5.4)

    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig2_identification():
    s = _s()["identification"]
    ms = ["DECIMATE", "SENDDELTA", "RF", "ARGUS"]
    x = np.arange(2); w = 0.2
    fig, ax = plt.subplots(1, 2, figsize=(CW * 2.02, 1.9))

    for i, m in enumerate(ms):
        ax[0].bar(i, s[m]["retention_pct"], 0.62, color=MCOL[m],
                  edgecolor="k", linewidth=0.4, hatch=HATCH[m],
                  yerr=s[m]["retention_ci95"], capsize=2, error_kw=dict(lw=0.6))
        ax[1].bar(i, s[m]["fidelity_err_pct"], 0.62, color=MCOL[m],
                  edgecolor="k", linewidth=0.4, hatch=HATCH[m],
                  label=MNAME[m])
    for a, ttl, yl in ((ax[0], "(a) records retained", "Retention (%)"),
                       (ax[1], "(b) reconstruction error", "Error (% of scale)")):
        a.set_xticks(range(len(ms)))
        a.set_xticklabels([MNAME[m] for m in ms], fontsize=6.0, rotation=12)
        a.set_title(ttl); a.set_ylabel(yl)
        a.grid(axis="y", alpha=0.25, lw=0.4)
    ax[0].set_ylim(50, 75)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{FIG}/fig2_identification.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig3_pareto():
    d = pd.read_csv(f"{RES}/tolerance_sweep.csv")
    g = d.groupby(["method", "mult"])[["retention_pct", "fidelity_err_pct"]].mean()
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for m, mk, c, lab in (("SENDDELTA", "s--", "#4c72b0", "send-on-delta"),
                          ("ARGUS", "o-", "#c0392b", "ARGUS")):
        q = g.loc[m].sort_values("retention_pct")
        ax.plot(q.retention_pct, q.fidelity_err_pct, mk, color=c, ms=3.2,
                label=lab, mfc="none" if m != "ARGUS" else c)
    for mu, off in ((0.5, (-20, 3)), (1.0, (4, 4)), (3.0, (4, 3))):
        q = g.loc["ARGUS"].loc[mu]
        ax.annotate(f"{mu}$\\epsilon$", (q.retention_pct, q.fidelity_err_pct),
                    textcoords="offset points", xytext=off, fontsize=5.4,
                    color="#c0392b")
    ax.set_xlabel("Records retained (%)")
    ax.set_ylabel("Reconstruction error (%)")
    ax.legend(loc="upper right", handlelength=1.8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig3_pareto.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig4_retrieval():
    s = _s()["retrieval"]
    eng = ["RAWSCAN", "TIMEIDX", "MATVIEW", "GRAIN"]
    qs = ["q1_ms", "q2_ms", "q3_ms", "q4_ms"]
    ql = ["Q1 latest\nevent", "Q2 events\nin window",
          "Q3 window\nmean, p90", "Q4 degree-\nhours"]
    x = np.arange(4); w = 0.2
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for i, e in enumerate(eng):
        ax.bar(x + (i - 1.5) * w, [s[e][q] for q in qs], w, label=ENAME[e],
               color=ECOL[e], edgecolor="k", linewidth=0.4, hatch=HATCH[e])
    ax.set_xticks(x); ax.set_xticklabels(ql, fontsize=6.0)
    ax.set_ylabel("Query latency (ms)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              handlelength=1.1, columnspacing=0.7, handletextpad=0.35,
              fontsize=5.6)
    ax.set_ylim(0, 0.075)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig4_retrieval.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig5_strategic():
    s = _s()["strategic"]
    ms = ["DECIMATE", "SENDDELTA", "RF", "ARGUS"]
    cols = ["vwc_err_pct", "stemp_err_pct", "degreehour_err_pct"]
    cl = ["mean water\ncontent", "mean soil\ntemperature", "accumulated\ndegree-hours"]
    x = np.arange(3); w = 0.2
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for i, m in enumerate(ms):
        ax.bar(x + (i - 1.5) * w, [s[m][c] for c in cols], w, label=MNAME[m],
               color=MCOL[m], edgecolor="k", linewidth=0.4, hatch=HATCH[m])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cl, fontsize=6.0)
    ax.set_ylabel("Error of season statistic (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05),
              handlelength=1.1, columnspacing=0.7, handletextpad=0.35,
              fontsize=5.6)
    ax.grid(axis="y", alpha=0.25, lw=0.4, which="both")
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig5_strategic.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1_architecture(); fig2_identification(); fig3_pareto()
    fig4_retrieval(); fig5_strategic()
    print("figures written to", FIG)
