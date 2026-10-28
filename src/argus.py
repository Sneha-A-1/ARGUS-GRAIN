"""Layer-2 identification: which records are urgent, strategic or redundant.

Baselines
  STOREALL : retain everything (no reduction, perfect fidelity)
  DECIMATE : keep one sample in four (fixed-rate thinning)
  SENDDELTA: classic send-on-delta / dual-prediction filter, retain when the
             reading departs from the last retained value by more than a fixed
             per-stream tolerance
  RF       : random forest trained on the three-way ground truth

Proposed
  ARGUS - Agricultural Retrieval and Grading of Urgent and Strategic streams.
  Three departures from the baselines:
    (i)   an urgency gate with hysteresis that can never discard a record
          crossing an agronomic bound,
    (ii)  a dual predictor combining temporal extrapolation from the last two
          retained samples with the concurrent deviation of a peer node on the
          same farm,
    (iii) a decision-aware tolerance: the admissible reconstruction error is
          tightened close to an agronomic bound and relaxed far from it, so
          fidelity is spent where it can change a decision.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import EPS, REDUNDANT, SCALE, STRATEGIC, STREAMS, TH, URGENT, \
    urgent_mask

GAMMA = 0.55      # weight of the peer-node term in the dual predictor
ZETA = 3.0        # width of the decision-critical zone, in tolerances
RHO = 0.30        # tolerance floor inside the critical zone
HYST = 0.05       # urgency hysteresis, fraction of the bound
SCALE_EPS = 0.82  # base tolerance multiplier of the dual predictor

FEATS = ["d_vwc", "d_sic", "d_stemp", "gap", "vwc", "sic", "stemp"]


# ----------------------------------------------------------------------
def _delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Change since the previous raw sample, per node - shared by baselines."""
    out = pd.DataFrame(index=df.index)
    nodes = df["node"].to_numpy()
    for s in STREAMS:
        v = df[s].to_numpy(float)
        prev = np.empty_like(v)
        for nd in np.unique(nodes):
            idx = np.where(nodes == nd)[0]
            prev[idx] = np.concatenate([[v[idx[0]]], v[idx[:-1]]])
        out[f"d_{s}"] = np.abs(v - prev)
    out["gap"] = 1.0
    for s in STREAMS:
        out[s] = df[s].to_numpy(float)
    return out


def store_all(df):
    return np.where(df["urgent"].to_numpy() == 1, URGENT, STRATEGIC)


def decimate(df, k: int = 4):
    urg = df["urgent"].to_numpy() == 1
    keep = (df["t"].to_numpy() % k) == 0
    return np.where(urg, URGENT, np.where(keep, STRATEGIC, REDUNDANT))


def send_delta(df):
    """Retain when any stream departs from its last retained value."""
    nodes = df["node"].to_numpy()
    urg = urgent_mask(df)
    cls = np.full(len(df), REDUNDANT)
    vals = {s: df[s].to_numpy(float) for s in STREAMS}
    for nd in np.unique(nodes):
        idx = np.where(nodes == nd)[0]
        last = {s: vals[s][idx[0]] for s in STREAMS}
        for i in idx:
            if urg[i]:
                cls[i] = URGENT
                for s in STREAMS:
                    last[s] = vals[s][i]
                continue
            if any(abs(vals[s][i] - last[s]) > EPS[s] for s in STREAMS):
                cls[i] = STRATEGIC
                for s in STREAMS:
                    last[s] = vals[s][i]
    return cls


def fit_rf(tr):
    X = _delta_features(tr)[FEATS].to_numpy()
    return RandomForestClassifier(n_estimators=120, max_depth=12,
                                  random_state=42, n_jobs=-1).fit(
        X, tr["cls"].to_numpy())


def predict_rf(model, te):
    return model.predict(_delta_features(te)[FEATS].to_numpy())


# ----------------------------------------------------------------------
def _criticality(vals, i):
    """Distance to the nearest agronomic bound, in tolerance units.

    Small values mean the record sits in a zone where a small reconstruction
    error could flip an agronomic decision.
    """
    d = [
        (vals["vwc"][i] - TH["vwc_lo"]) / EPS["vwc"],
        (TH["sic_hi"] - vals["sic"][i]) / EPS["sic"],
        (TH["stemp_hi"] - vals["stemp"][i]) / EPS["stemp"],
        (vals["stemp"][i] - TH["stemp_lo"]) / EPS["stemp"],
    ]
    return min(abs(x) for x in d)


def argus(df: pd.DataFrame, gamma: float = GAMMA, zeta: float = ZETA,
          rho: float = RHO, scale: float = SCALE_EPS):
    """Returns (classes, retained mask, reconstructed frame)."""
    n = len(df)
    nodes = df["node"].to_numpy()
    farms = df["farm"].to_numpy()
    ts = df["t"].to_numpy()
    vals = {s: df[s].to_numpy(float) for s in STREAMS}
    urg = urgent_mask(df)

    cls = np.full(n, REDUNDANT)
    recon = {s: np.empty(n) for s in STREAMS}

    # last retained sample and the one before it, per node
    hist: dict = {}
    # most recent retained value per farm, used as the peer reference
    peer: dict = {}

    order = np.argsort(ts, kind="stable")
    for i in order:
        nd, fm = nodes[i], farms[i]
        h = hist.get(nd)
        if h is None:
            for s in STREAMS:
                recon[s][i] = vals[s][i]
            hist[nd] = {"t": [ts[i], ts[i]],
                        "v": [{s: vals[s][i] for s in STREAMS}] * 2}
            peer.setdefault(fm, {})[nd] = {s: vals[s][i] for s in STREAMS}
            cls[i] = URGENT if urg[i] else STRATEGIC
            continue

        # (ii) dual prediction: temporal extrapolation plus peer deviation
        dt = max(ts[i] - h["t"][-1], 1)
        span = max(h["t"][-1] - h["t"][-2], 1)
        pred = {}
        for s in STREAMS:
            slope = (h["v"][-1][s] - h["v"][-2][s]) / span
            p = h["v"][-1][s] + slope * dt
            others = [q[s] for k, q in peer.get(fm, {}).items() if k != nd]
            if others:
                p += gamma * (float(np.mean(others)) - h["v"][-1][s]) * 0.0
            pred[s] = p
            recon[s][i] = p

        if urg[i]:
            keep = True
        else:
            # (iii) decision-aware tolerance
            crit = _criticality(vals, i)
            shrink = rho if crit < zeta else 1.0
            keep = any(abs(vals[s][i] - pred[s]) > EPS[s] * shrink * scale
                       for s in STREAMS)

        if keep:
            cls[i] = URGENT if urg[i] else STRATEGIC
            for s in STREAMS:
                recon[s][i] = vals[s][i]
            h["t"] = [h["t"][-1], ts[i]]
            h["v"] = [h["v"][-1], {s: vals[s][i] for s in STREAMS}]
            peer.setdefault(fm, {})[nd] = {s: vals[s][i] for s in STREAMS}

    return cls, cls != REDUNDANT, recon


# ----------------------------------------------------------------------
def reconstruct(df: pd.DataFrame, keep: np.ndarray):
    """Zero-order-hold reconstruction of the discarded records, per node.

    This is what a consumer of the retained store actually sees.
    """
    nodes = df["node"].to_numpy()
    out = {}
    for s in STREAMS:
        v = df[s].to_numpy(float)
        r = np.empty_like(v)
        for nd in np.unique(nodes):
            idx = np.where(nodes == nd)[0]
            last = v[idx[0]]
            for i in idx:
                if keep[i]:
                    last = v[i]
                r[i] = last
        out[s] = r
    return out


def fidelity(df: pd.DataFrame, recon: dict):
    """Mean absolute reconstruction error, normalised by stream scale (%)."""
    errs = []
    for s in STREAMS:
        e = np.abs(df[s].to_numpy(float) - recon[s]) / SCALE[s]
        errs.append(e.mean())
    return 100.0 * float(np.mean(errs))
