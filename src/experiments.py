"""Evaluation of ARGUS + GRAIN on the NEON Konza soil array.

Repetitions come from five contiguous temporal folds of the held-out period
rather than random seeds, since the data are a single real deployment.
"""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))

import argus as A
import grain as G
from data import (DT_MIN, REDUNDANT, STRATEGIC, STREAMS, URGENT, load_neon,
                  split)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
NFOLD = 10
IDENT = ["STOREALL", "DECIMATE", "SENDDELTA", "RF", "ARGUS"]
ENGINES = ["RAWSCAN", "TIMEIDX", "MATVIEW", "GRAIN"]
_CACHE = {}


def data():
    if "df" not in _CACHE:
        df = load_neon()
        tr, va, te = split(df)
        _CACHE.update(df=df, tr=tr, va=va, te=te, rf=A.fit_rf(tr))
    return _CACHE


def folds(te, n=NFOLD):
    edges = np.linspace(te["t"].min(), te["t"].max() + 1, n + 1).astype(int)
    for k in range(n):
        m = (te["t"] >= edges[k]) & (te["t"] < edges[k + 1])
        yield k, te[m].reset_index(drop=True)


def macro_f1(y, p):
    fs = []
    for c in (URGENT, STRATEGIC, REDUNDANT):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs)), fs


def ci95(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def identification():
    d = data(); rows = []
    for k, te in folds(d["te"]):
        y = te["cls"].to_numpy(); urg = te["urgent"].to_numpy() == 1
        timed = {}
        t0 = time.perf_counter(); p_sa = A.store_all(te); timed["STOREALL"] = time.perf_counter() - t0
        t0 = time.perf_counter(); p_de = A.decimate(te); timed["DECIMATE"] = time.perf_counter() - t0
        t0 = time.perf_counter(); p_sd = A.send_delta(te); timed["SENDDELTA"] = time.perf_counter() - t0
        t0 = time.perf_counter(); p_rf = A.predict_rf(d["rf"], te); timed["RF"] = time.perf_counter() - t0
        t0 = time.perf_counter(); p_ar, keep_ar, rec_ar = A.argus(te); timed["ARGUS"] = time.perf_counter() - t0

        for name, p in (("STOREALL", p_sa), ("DECIMATE", p_de),
                        ("SENDDELTA", p_sd), ("RF", p_rf), ("ARGUS", p_ar)):
            keep = p != REDUNDANT
            rec = rec_ar if name == "ARGUS" else A.reconstruct(te, keep)
            f1, per = macro_f1(y, p)
            rows.append(dict(fold=k, method=name, macro_f1=f1,
                             f1_urgent=per[0], f1_strategic=per[1],
                             f1_redundant=per[2],
                             urgent_recall=100.0 * float((p[urg] == URGENT).mean()),
                             retention_pct=100.0 * float(keep.mean()),
                             fidelity_err_pct=A.fidelity(te, rec),
                             throughput_kps=len(te) / max(timed[name], 1e-9) / 1000.0))
    return pd.DataFrame(rows)


def strategic_accuracy():
    d = data(); rows = []
    for k, te in folds(d["te"]):
        sites = sorted(te.farm.unique())

        def stats(vals):
            out = {}
            for f in sites:
                m = te["farm"].to_numpy() == f
                out[f] = (float(vals["vwc"][m].mean()),
                          float(vals["stemp"][m].mean()),
                          float(np.clip(vals["stemp"][m] - 15.0, 0, None).sum()
                                * DT_MIN / 60.0),
                          float(np.percentile(vals["vwc"][m], 10)))
            return out

        ref = stats({s: te[s].to_numpy(float) for s in STREAMS})
        p_ar, keep_ar, rec_ar = A.argus(te)
        for name, p, rec in (("DECIMATE", A.decimate(te), None),
                             ("SENDDELTA", A.send_delta(te), None),
                             ("RF", A.predict_rf(d["rf"], te), None),
                             ("ARGUS", p_ar, rec_ar)):
            keep = p != REDUNDANT
            vals = rec if rec is not None else A.reconstruct(te, keep)
            got = stats(vals)
            e = lambda j: 100.0 * float(np.mean(
                [abs(got[f][j] - ref[f][j]) / max(abs(ref[f][j]), 1e-9)
                 for f in sites]))
            rows.append(dict(fold=k, method=name, vwc_err_pct=e(0),
                             stemp_err_pct=e(1), degreehour_err_pct=e(2),
                             p10_err_pct=e(3)))
    return pd.DataFrame(rows)


def retrieval(n_queries=300):
    d = data(); rows = []
    for k, te in folds(d["te"]):
        keep = A.argus(te)[1]
        rng = np.random.default_rng(100 + k)
        nodes = te["node"].unique(); sites = te["farm"].unique()
        tmin, tmax = int(te["t"].min()), int(te["t"].max())
        for cls in ENGINES:
            eng = {"RAWSCAN": G.RawScan, "TIMEIDX": G.TimeIndex,
                   "MATVIEW": G.MatView, "GRAIN": G.Grain}[cls]
            t0 = time.perf_counter(); e = eng(te, keep)
            build = time.perf_counter() - t0
            lat = {}
            t0 = time.perf_counter()
            for _ in range(n_queries):
                e.q1_latest_event(int(rng.choice(nodes)), str(rng.choice(G.EVENTS)))
            lat["q1_ms"] = 1000.0 * (time.perf_counter() - t0) / n_queries
            t0 = time.perf_counter()
            for _ in range(n_queries):
                a = int(rng.integers(tmin, max(tmin + 1, tmax - 400)))
                e.q2_events_window(int(rng.choice(sites)),
                                   str(rng.choice(G.EVENTS)), a, a + 336)
            lat["q2_ms"] = 1000.0 * (time.perf_counter() - t0) / n_queries
            t0 = time.perf_counter()
            for _ in range(n_queries):
                a = int(rng.integers(tmin, max(tmin + 1, tmax - 400)))
                e.q3_window_vwc(int(rng.choice(sites)), a, a + 336)
            lat["q3_ms"] = 1000.0 * (time.perf_counter() - t0) / n_queries
            nq4 = max(n_queries // 4, 20)
            t0 = time.perf_counter()
            for _ in range(nq4):
                a = int(rng.integers(tmin, max(tmin + 1, tmax - 700)))
                e.q4_gdd(int(rng.choice(sites)), a, a + 672)
            lat["q4_ms"] = 1000.0 * (time.perf_counter() - t0) / nq4
            rows.append(dict(fold=k, engine=cls, build_s=build,
                             footprint_mb=e.footprint_mb(), **lat))
    return pd.DataFrame(rows)


def tolerance_sweep():
    import data as D
    d = data(); rows = []
    base = dict(D.EPS)
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for k, te in folds(d["te"], 5):
            try:
                for key in base:
                    D.EPS[key] = base[key] * mult
                p_ar, keep_ar, rec_ar = A.argus(te)
                for name, p, rec in (("SENDDELTA", A.send_delta(te), None),
                                     ("ARGUS", p_ar, rec_ar)):
                    keep = p != REDUNDANT
                    vals = rec if rec is not None else A.reconstruct(te, keep)
                    rows.append(dict(mult=mult, fold=k, method=name,
                                     retention_pct=100.0 * float(keep.mean()),
                                     fidelity_err_pct=A.fidelity(te, vals)))
            finally:
                for key in base:
                    D.EPS[key] = base[key]
    return pd.DataFrame(rows)


def main():
    os.makedirs(RES, exist_ok=True)
    idf = identification(); idf.to_csv(f"{RES}/identification_raw.csv", index=False)
    sad = strategic_accuracy(); sad.to_csv(f"{RES}/strategic_raw.csv", index=False)
    ret = retrieval(); ret.to_csv(f"{RES}/retrieval_raw.csv", index=False)
    swp = tolerance_sweep(); swp.to_csv(f"{RES}/tolerance_sweep.csv", index=False)

    out = {"identification": {}, "strategic": {}, "retrieval": {}}
    for m, v in idf.groupby("method"):
        out["identification"][m] = {c: round(float(v[c].mean()), 4) for c in
                                    ("macro_f1", "f1_urgent", "f1_strategic",
                                     "f1_redundant", "urgent_recall",
                                     "retention_pct", "fidelity_err_pct",
                                     "throughput_kps")}
        out["identification"][m]["retention_ci95"] = round(float(ci95(v.retention_pct)), 3)
    for m, v in sad.groupby("method"):
        out["strategic"][m] = {c: round(float(v[c].mean()), 4) for c in
                               ("vwc_err_pct", "stemp_err_pct",
                                "degreehour_err_pct", "p10_err_pct")}
    for m, v in ret.groupby("engine"):
        out["retrieval"][m] = {c: round(float(v[c].mean()), 4) for c in
                               ("q1_ms", "q2_ms", "q3_ms", "q4_ms",
                                "build_s", "footprint_mb")}
    out["tolerance_sweep"] = {
        str(mu): {m: {"retention_pct": round(float(g.retention_pct.mean()), 2),
                      "fidelity_err_pct": round(float(g.fidelity_err_pct.mean()), 4)}
                  for m, g in swp[swp.mult == mu].groupby("method")}
        for mu in sorted(swp.mult.unique())}

    def w(a, b, col, frame, key):
        return float(wilcoxon(frame[frame[key] == a][col].to_numpy(),
                              frame[frame[key] == b][col].to_numpy()).pvalue)
    out["wilcoxon"] = {
        "fidelity_argus_vs_senddelta": w("ARGUS", "SENDDELTA", "fidelity_err_pct", idf, "method"),
        "fidelity_argus_vs_rf": w("ARGUS", "RF", "fidelity_err_pct", idf, "method"),
        "q1_grain_vs_timeidx": w("GRAIN", "TIMEIDX", "q1_ms", ret, "engine"),
        "q3_grain_vs_timeidx": w("GRAIN", "TIMEIDX", "q3_ms", ret, "engine"),
    }

    d = data(); df0 = d["df"]
    out["dataset"] = {
        "records": int(len(df0)), "nodes": int(df0.node.nunique()),
        "plots": int(df0.farm.nunique()), "depths": int(df0.depth.nunique()),
        "timesteps": int(df0.t.nunique()),
        "days": round(float(df0.t.max() * DT_MIN / 60 / 24), 1),
        "test_records": int(len(d["te"])),
        "urgent_pct": round(100 * float((df0.cls == URGENT).mean()), 2),
        "strategic_pct": round(100 * float((df0.cls == STRATEGIC).mean()), 2),
        "redundant_pct": round(100 * float((df0.cls == REDUNDANT).mean()), 2),
        "qf_pass_pct": round(100 * float((df0.qf == 0).mean()), 1),
        "vwc_mean": round(float(df0.vwc.mean()), 3),
        "sic_mean": round(float(df0.sic.mean()), 1),
        "stemp_mean": round(float(df0.stemp.mean()), 2),
    }
    with open(f"{RES}/summary.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    s = main()
    print(json.dumps(s["identification"], indent=1))
    print(json.dumps(s["retrieval"], indent=1))
    print(json.dumps(s["strategic"], indent=1))
    print(json.dumps(s["dataset"], indent=1))
