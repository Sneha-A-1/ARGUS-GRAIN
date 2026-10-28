"""Layer-3 retrieval: serving urgent and strategic queries over retained data.

Query classes
  Q1 latest urgent event of a given type for one node        (point)
  Q2 all urgent events of a given type on one farm in a window (range)
  Q3 monthly mean and p90 of root-zone water content per farm  (aggregate)
  Q4 accumulated growing degree days per farm over the season  (aggregate)

Engines
  RAWSCAN  : linear scan of the retained records
  TIMEIDX  : records sorted by timestamp, binary search then scan
  MATVIEW  : precomputed monthly aggregates only; urgent queries fall back to
             a linear scan
  GRAIN    : proposed - Grouped Retrieval with Aggregation and INcremental
             indexing. One ingestion pass maintains (a) an inverted index from
             (node, event type) to timestamps, (b) per-farm per-day sketches
             holding count, sum, sum of squares, min, max and a 16-bin
             histogram for every stream. Both query classes are then answered
             from summaries rather than from the record store.
"""

from __future__ import annotations
import bisect
import numpy as np
import pandas as pd

from data import DT_MIN, STREAMS, TH

EVENTS = ("dry", "saline", "hot", "cold")
NBINS = 16
GDD_BASE = 15.0                      # degC, base for soil degree-hours


def event_flags(row_vals, i):
    return {
        "dry": row_vals["vwc"][i] < TH["vwc_lo"],
        "saline": row_vals["sic"][i] > TH["sic_hi"],
        "hot": row_vals["stemp"][i] > TH["stemp_hi"],
        "cold": row_vals["stemp"][i] < TH["stemp_lo"],
    }


# ----------------------------------------------------------------------
class RawScan:
    """Retained records held as flat arrays; every query is a scan."""

    name = "RAWSCAN"

    def __init__(self, df: pd.DataFrame, keep: np.ndarray):
        d = df[keep]
        self.t = d["t"].to_numpy()
        self.node = d["node"].to_numpy()
        self.farm = d["farm"].to_numpy()
        self.v = {s: d[s].to_numpy(float) for s in STREAMS}

    def footprint_mb(self):
        return (self.t.nbytes + self.node.nbytes + self.farm.nbytes
                + sum(a.nbytes for a in self.v.values())) / 1e6

    def _ev(self, kind):
        if kind == "dry":
            return self.v["vwc"] < TH["vwc_lo"]
        if kind == "saline":
            return self.v["sic"] > TH["sic_hi"]
        if kind == "hot":
            return self.v["stemp"] > TH["stemp_hi"]
        return self.v["stemp"] < TH["stemp_lo"]

    def q1_latest_event(self, node, kind):
        m = (self.node == node) & self._ev(kind)
        return int(self.t[m].max()) if m.any() else -1

    def q2_events_window(self, farm, kind, t0, t1):
        m = (self.farm == farm) & self._ev(kind) & (self.t >= t0) & (self.t <= t1)
        return int(m.sum())

    def q3_window_vwc(self, farm, t0, t1):
        m = (self.farm == farm) & (self.t >= t0) & (self.t <= t1)
        x = self.v["vwc"][m]
        return (float(x.mean()), float(np.percentile(x, 90))) if x.size \
            else (float("nan"), float("nan"))

    def q4_gdd(self, farm, t0=None, t1=None):
        m = self.farm == farm
        if t0 is not None:
            m &= (self.t >= t0) & (self.t <= t1)
        x = self.v["stemp"][m]
        return float(np.clip(x - GDD_BASE, 0, None).sum() * DT_MIN / 1440.0)


class TimeIndex(RawScan):
    """Timestamp-sorted store with binary search on the range predicate."""

    name = "TIMEIDX"

    def __init__(self, df, keep):
        super().__init__(df, keep)
        o = np.argsort(self.t, kind="stable")
        self.t = self.t[o]; self.node = self.node[o]; self.farm = self.farm[o]
        self.v = {s: a[o] for s, a in self.v.items()}
        self.tl = self.t.tolist()

    def _slice(self, t0, t1):
        return bisect.bisect_left(self.tl, t0), bisect.bisect_right(self.tl, t1)

    def q2_events_window(self, farm, kind, t0, t1):
        a, b = self._slice(t0, t1)
        m = (self.farm[a:b] == farm) & self._ev(kind)[a:b]
        return int(m.sum())

    def q3_window_vwc(self, farm, t0, t1):
        a, b = self._slice(t0, t1)
        m = self.farm[a:b] == farm
        x = self.v["vwc"][a:b][m]
        return (float(x.mean()), float(np.percentile(x, 90))) if x.size \
            else (float("nan"), float("nan"))


class MatView(TimeIndex):
    """Fixed monthly materialised views; event queries still scan."""

    name = "MATVIEW"

    def __init__(self, df, keep):
        super().__init__(df, keep)
        per = int(30 * 24 * 60 / DT_MIN)
        self.view = {}
        self.gdd = {}
        for fm in np.unique(self.farm):
            mf = self.farm == fm
            self.gdd[int(fm)] = float(np.clip(
                self.v["stemp"][mf] - GDD_BASE, 0, None).sum()
                * DT_MIN / 1440.0)
            for mo in range(int(self.t.max() // per) + 1):
                m = mf & (self.t >= mo * per) & (self.t < (mo + 1) * per)
                x = self.v["vwc"][m]
                if x.size:
                    self.view[(int(fm), mo)] = (float(x.mean()),
                                                float(np.percentile(x, 90)))

    def footprint_mb(self):
        return super().footprint_mb() + len(self.view) * 24 / 1e6

    def q3_window_vwc(self, farm, t0, t1):
        per = int(30 * 24 * 60 / DT_MIN)
        if t0 % per == 0 and (t1 + 1) % per == 0:          # view hit
            return self.view.get((int(farm), t0 // per),
                                 (float("nan"), float("nan")))
        return TimeIndex.q3_window_vwc(self, farm, t0, t1)  # view miss: scan

    def q4_gdd(self, farm, t0=None, t1=None):
        if t0 is None:
            return self.gdd.get(int(farm), float("nan"))
        return TimeIndex.q4_gdd(self, farm, t0, t1)


# ----------------------------------------------------------------------
class Grain(RawScan):
    """Proposed engine: inverted event index plus per-farm daily sketches."""

    name = "GRAIN"

    def __init__(self, df: pd.DataFrame, keep: np.ndarray):
        d = df[keep]
        t = d["t"].to_numpy(); node = d["node"].to_numpy()
        farm = d["farm"].to_numpy()
        v = {s: d[s].to_numpy(float) for s in STREAMS}
        self.t, self.node, self.farm, self.v = t, node, farm, v

        per_day = int(24 * 60 / DT_MIN)
        self.index: dict = {}          # (node, kind) -> sorted timestamps
        self.findex: dict = {}         # (farm, kind) -> sorted timestamps
        self.sk: dict = {}             # (farm, day) -> per-stream sketch
        self.edges = {s: None for s in STREAMS}
        for s in STREAMS:
            lo, hi = float(v[s].min()), float(v[s].max())
            self.edges[s] = np.linspace(lo, hi + 1e-9, NBINS + 1)

        flags = {k: None for k in EVENTS}
        flags["dry"] = v["vwc"] < TH["vwc_lo"]
        flags["saline"] = v["sic"] > TH["sic_hi"]
        flags["hot"] = v["stemp"] > TH["stemp_hi"]
        flags["cold"] = v["stemp"] < TH["stemp_lo"]
        

        order = np.argsort(t, kind="stable")
        for i in order:                                  # single ingest pass
            for k in EVENTS:
                if flags[k][i]:
                    self.index.setdefault((int(node[i]), k), []).append(int(t[i]))
                    self.findex.setdefault((int(farm[i]), k), []).append(int(t[i]))
            key = (int(farm[i]), int(t[i]) // per_day)
            e = self.sk.get(key)
            if e is None:
                e = self.sk[key] = {s: [0, 0.0, 0.0, np.inf, -np.inf,
                                        np.zeros(NBINS, np.int32)]
                                    for s in STREAMS}
            for s in STREAMS:
                x = v[s][i]
                a = e[s]
                a[0] += 1; a[1] += x; a[2] += x * x
                if x < a[3]:
                    a[3] = x
                if x > a[4]:
                    a[4] = x
                b = int(np.searchsorted(self.edges[s], x, side="right") - 1)
                a[5][min(max(b, 0), NBINS - 1)] += 1

    def footprint_mb(self):
        base = (self.t.nbytes + self.node.nbytes + self.farm.nbytes
                + sum(a.nbytes for a in self.v.values()))
        idx = sum(len(x) for x in self.index.values()) * 8
        idx += sum(len(x) for x in self.findex.values()) * 8
        sk = len(self.sk) * len(STREAMS) * (5 * 8 + NBINS * 4)
        return (base + idx + sk) / 1e6

    def q1_latest_event(self, node, kind):
        lst = self.index.get((int(node), kind))
        return lst[-1] if lst else -1

    def q2_events_window(self, farm, kind, t0, t1):
        lst = self.findex.get((int(farm), kind))
        if not lst:
            return 0
        return bisect.bisect_right(lst, t1) - bisect.bisect_left(lst, t0)

    def q3_window_vwc(self, farm, t0, t1):
        per_day = int(24 * 60 / DT_MIN)
        d0, d1 = int(t0) // per_day, int(t1) // per_day
        cnt = 0; ssum = 0.0
        hist = np.zeros(NBINS, np.int64)
        for day in range(d0, d1 + 1):
            e = self.sk.get((int(farm), day))
            if e is None:
                continue
            a = e["vwc"]
            cnt += a[0]; ssum += a[1]; hist += a[5]
        if not cnt:
            return (float("nan"), float("nan"))
        target = 0.90 * cnt
        c = 0
        edges = self.edges["vwc"]
        p90 = edges[-1]
        for b in range(NBINS):
            c += hist[b]
            if c >= target:
                p90 = 0.5 * (edges[b] + edges[b + 1])
                break
        return (ssum / cnt, float(p90))

    def q4_gdd(self, farm, t0=None, t1=None):
        per_day = int(24 * 60 / DT_MIN)
        d0 = None if t0 is None else int(t0) // per_day
        d1 = None if t1 is None else int(t1) // per_day
        tot = 0.0
        for (fm, day), e in self.sk.items():
            if fm != int(farm) or (d0 is not None and not d0 <= day <= d1):
                continue
            a = e["stemp"]
            tot += max(a[1] / max(a[0], 1) - GDD_BASE, 0.0) * a[0]
        return float(tot * DT_MIN / 1440.0)
