"""Sensor-stream construction from the NEON Konza Prairie soil array.

Primary dataset
    National Ecological Observatory Network (NEON), Konza Prairie
    Agricultural Site (KONA), Kansas, 1 May - 31 August 2023.
      DP1.00094.001  Soil water content and water salinity (30 min)
      DP1.00041.001  Soil temperature (30 min)
    Downloaded with the `neonutilities` package and stacked into
    SWS_30_minute.csv and ST_30_minute.csv.

Deployment model
    The site instruments five soil plots, each at eight measurement depths.
    Each plot is treated as a *site* and each (plot, depth) pair as a *sensor
    node* carrying three streams: volumetric water content (m3/m3), soil ion
    content (uS/cm, a salinity proxy) and soil temperature (degC). Nodes in
    the same plot are physically co-located, which is what makes the
    peer-prediction term of ARGUS meaningful.

    The records are used as released: about 15 % of raw rows carry no value
    and only 60 % pass the NEON quality flag. Gaps and flagged samples are
    part of the workload a retrieval framework must survive, so they are
    retained rather than repaired.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DT_MIN = 30.0                                  # native sampling interval
STREAMS = ("vwc", "sic", "stemp")
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# agronomic decision bounds, set from the measured distributions of this site
TH = {
    "vwc_lo": 0.10,       # m3/m3   root-zone water deficit  (18 % of records)
    "sic_hi": 1800.0,     # uS/cm   salinity build-up        (10 %)
    "stemp_hi": 30.0,     # degC    root-zone heat stress    ( 5 %)
    "stemp_lo": 14.0,     # degC    cold soil, slow uptake   ( 3 %)
}

# retrieval tolerance per stream, in physical units
EPS = {"vwc": 0.004, "sic": 12.0, "stemp": 0.20}

# stationary scale used to normalise reconstruction error
SCALE = {"vwc": 0.136, "sic": 396.0, "stemp": 5.13}

URGENT, STRATEGIC, REDUNDANT = 0, 1, 2


# ----------------------------------------------------------------------
def load_neon(sws_csv: str = None, st_csv: str = None) -> pd.DataFrame:
    """Join the two NEON products into one long frame of node-time records."""
    sws_csv = sws_csv or os.path.join(RAW, "SWS_30_minute.csv")
    st_csv = st_csv or os.path.join(RAW, "ST_30_minute.csv")
    sws = pd.read_csv(sws_csv, low_memory=False)
    st = pd.read_csv(st_csv, low_memory=False)

    m = sws.merge(st, on=["startDateTime", "horizontalPosition",
                          "verticalPosition"], how="inner")
    m = m.dropna(subset=["VSWCMean", "soilTempMean"]).copy()
    m["VSICMean"] = m["VSICMean"].ffill().bfill()

    m["ts"] = pd.to_datetime(m["startDateTime"])
    t0 = m["ts"].min()
    m["t"] = ((m["ts"] - t0).dt.total_seconds()
              / (60 * DT_MIN)).round().astype(int)
    m["site"] = m["horizontalPosition"].astype(int)
    m["depth"] = m["verticalPosition"].astype(int)
    m["node"] = m["site"] * 100 + (m["depth"] - 500)

    df = pd.DataFrame({
        "t": m["t"].to_numpy(),
        "node": m["node"].to_numpy(),
        "farm": m["site"].to_numpy(),          # plot plays the role of a site
        "depth": m["depth"].to_numpy(),
        "vwc": m["VSWCMean"].to_numpy(float),
        "sic": m["VSICMean"].to_numpy(float),
        "stemp": m["soilTempMean"].to_numpy(float),
        "qf": m["VSWCFinalQF"].fillna(1).to_numpy(int),
    }).sort_values(["t", "node"]).reset_index(drop=True)
    return label(df)


def synthesize(*args, **kwargs):
    """Kept for API compatibility; the evaluation now uses the NEON records."""
    raise FileNotFoundError(
        "NEON CSVs not found in data/raw/. Download DP1.00094.001 and "
        "DP1.00041.001 for site KONA (2023-05 to 2023-08); see data/README.md.")


# ----------------------------------------------------------------------
def urgency(vwc, sic, stemp, margin=0.0):
    vwc = np.asarray(vwc, float); sic = np.asarray(sic, float)
    stemp = np.asarray(stemp, float)
    return ((vwc < TH["vwc_lo"] * (1.0 + margin)) |
            (sic > TH["sic_hi"] * (1.0 - margin)) |
            (stemp > TH["stemp_hi"] * (1.0 - margin)) |
            (stemp < TH["stemp_lo"] * (1.0 + margin)))


def urgency_abs(vwc, sic, stemp, g_vwc=0.0, g_sic=0.0, g_stemp=0.0):
    """Urgency predicate with per-stream absolute guard bands."""
    return ((np.asarray(vwc, float) < TH["vwc_lo"] + g_vwc) |
            (np.asarray(sic, float) > TH["sic_hi"] - g_sic) |
            (np.asarray(stemp, float) > TH["stemp_hi"] - g_stemp) |
            (np.asarray(stemp, float) < TH["stemp_lo"] + g_stemp))


def urgent_mask(df: pd.DataFrame) -> np.ndarray:
    return urgency(df["vwc"], df["sic"], df["stemp"])


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Three-way ground truth.

    URGENT    : an agronomic bound is crossed.
    STRATEGIC : not urgent, but the record differs from the previous sample of
                the same node by more than the retrieval tolerance.
    REDUNDANT : not urgent and reconstructable within tolerance.
    """
    df = df.copy()
    urg = urgent_mask(df)
    nodes = df["node"].to_numpy()
    informative = np.zeros(len(df), bool)
    for s in STREAMS:
        v = df[s].to_numpy(float)
        prev = np.empty_like(v)
        for nd in np.unique(nodes):
            idx = np.where(nodes == nd)[0]
            prev[idx] = np.concatenate([[v[idx[0]]], v[idx[:-1]]])
        informative |= np.abs(v - prev) > EPS[s]
    df["cls"] = np.where(urg, URGENT,
                         np.where(informative, STRATEGIC, REDUNDANT))
    df["urgent"] = urg.astype(int)
    return df


def split(df: pd.DataFrame):
    """Chronological 70/15/15 split."""
    tmax = df["t"].max() + 1
    a, b = int(0.70 * tmax), int(0.85 * tmax)
    return (df[df["t"] < a].reset_index(drop=True),
            df[(df["t"] >= a) & (df["t"] < b)].reset_index(drop=True),
            df[df["t"] >= b].reset_index(drop=True))
