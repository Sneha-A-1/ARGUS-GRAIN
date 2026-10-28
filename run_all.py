#!/usr/bin/env python3
"""Reproduce every table and figure of the ARGUS + GRAIN paper."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import experiments, figures

if __name__ == "__main__":
    print("[1/2] identification, strategic accuracy, retrieval, sweep ...")
    s = experiments.main()
    print("[2/2] figures ...")
    figures.fig1_architecture(); figures.fig2_identification()
    figures.fig3_pareto(); figures.fig4_retrieval(); figures.fig5_strategic()
    i, r = s["identification"], s["retrieval"]
    print("\ndone. results/ and figures/ updated.")
    print("  retention  send-on-delta %.2f%% -> ARGUS %.2f%%"
          % (i["SENDDELTA"]["retention_pct"], i["ARGUS"]["retention_pct"]))
    print("  recon err  send-on-delta %.3f%% -> ARGUS %.3f%%"
          % (i["SENDDELTA"]["fidelity_err_pct"], i["ARGUS"]["fidelity_err_pct"]))
    print("  Q3 latency time-index %.4f ms -> GRAIN %.4f ms"
          % (r["TIMEIDX"]["q3_ms"], r["GRAIN"]["q3_ms"]))
