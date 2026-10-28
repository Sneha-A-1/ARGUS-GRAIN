# ARGUS + GRAIN: an agricultural data retrieval framework

Reference implementation for *"An Agricultural Data Retrieval Framework for
Identifying and Processing IoT Streams into Urgent and Strategic Insights"*.

A soil sensor array has to answer two different questions. **Urgent**: is any
node in trouble right now, answered in milliseconds. **Strategic**: how did
water content evolve across the season, answered faithfully from the archive.
This repository implements a framework that serves both from one reduced
store.

* **ARGUS** - Agricultural Retrieval and Grading of Urgent and Strategic
  streams. Grades every record as urgent, strategic or redundant using a dual
  predictor (temporal extrapolation + peer nodes in the same plot) and a
  **decision-aware tolerance** that tightens admissible reconstruction error
  near an agronomic bound and relaxes it far from one. An urgency gate runs
  first, so discarding an urgent record is structurally impossible.
* **GRAIN** - Grouped Retrieval with Aggregation and INcremental indexing.
  One ingestion pass builds an inverted event index and per-plot daily
  sketches (count, sum, sum of squares, min, max, 16-bin histogram), so urgent
  lookups and ad-hoc seasonal aggregates are answered from summaries instead
  of scans.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # ~4 min, data included
```

Results land in `results/` (CSV + `summary.json`) and `figures/` (300 dpi
PNG). Every number in the paper comes from this script.

## Layout

| path | contents |
|---|---|
| `src/data.py` | NEON loader, deployment model, three-way ground truth |
| `src/argus.py` | ARGUS plus store-all, decimate, send-on-delta and random-forest baselines |
| `src/grain.py` | retrieval engines: raw scan, time index, materialised view, GRAIN |
| `src/experiments.py` | identification, strategic accuracy, retrieval, tolerance sweep, Wilcoxon |
| `src/figures.py` | Figs. 1-5 of the paper |
| `data/raw/` | the two NEON CSVs (included) |

## Data

NEON Konza Prairie Agricultural Site (KONA), Kansas, 1 May - 31 Aug 2023:
DP1.00094.001 (soil water content and salinity) and DP1.00041.001 (soil
temperature), 30-minute resolution, from <https://data.neonscience.org>.
Joining them gives **185,711 records across 38 nodes** (5 plots x 8 depths)
and 5,090 time steps. See `data/README.md` for re-download instructions.

Used as released: ~15 % of raw rows are empty and only 60.3 % pass the NEON
quality flag. Neither is repaired.

## Results (10 temporal folds of the held-out period)

Identification, 27,250 held-out records:

| Method | Macro F1 | Urgent recall | Retention | Recon. error | Throughput |
|---|---|---|---|---|---|
| Store-all | 0.364 | 100 % | 100.00 % | 0.000 % | — |
| Decimate 1-in-4 | 0.639 | 100 % | 68.24 % | 0.529 % | 53,209 k/s |
| Send-on-delta | 0.859 | 100 % | 61.91 % | 0.297 % | 1,332 k/s |
| Random forest | **0.998** | 100 % | **59.61 %** | 0.995 % | 173 k/s |
| **ARGUS** | 0.748 | 100 % | 62.58 % | **0.210 %** | 55 k/s |

ARGUS cuts reconstruction error 29.3 % below send-on-delta at equal retention
and 78.9 % below the random forest. The random forest's macro-F1 is near
tautological — it is trained on exactly those labels — and it is
simultaneously the worst method at reconstructing what it discarded, which is
the paper's methodological point.

Query latency (ms), ad-hoc windows:

| Engine | Q1 latest event | Q2 events in window | Q3 window mean/p90 | Q4 degree-hours | Build (s) | Size (MB) |
|---|---|---|---|---|---|---|
| Raw scan | 0.0146 | 0.0181 | 0.0542 | 0.0170 | 0.0003 | 0.082 |
| Time index | 0.0146 | 0.0167 | 0.0524 | 0.0171 | 0.0004 | 0.082 |
| Mat. view | 0.0144 | 0.0165 | 0.0523 | 0.0160 | 0.0008 | 0.082 |
| **GRAIN** | **0.0104** | **0.0116** | **0.0140** | **0.0081** | 0.0120 | 0.119 |

Wilcoxon signed-rank: p = 0.002 for ARGUS vs send-on-delta and vs random
forest on reconstruction error, and for GRAIN vs time index on Q1 and Q3.

Two honest costs: GRAIN's ingestion pass is 30x slower than building a sorted
index (12.0 ms vs 0.4 ms, paid once per batch), and its footprint is 45 %
larger. Decimation also beats ARGUS on soil-temperature statistics (0.012 %
vs 0.021 %), because temperature at depth is smooth and periodic while ARGUS
spends its budget on the water and salinity streams where the agronomic
bounds live.

## Citation

Sneha, P. D. Singh, V. Tripathi, "An Agricultural Data Retrieval
Framework for Identifying and Processing IoT Streams into Urgent and
Strategic Insights" (submitted).

## License

MIT — see `LICENSE`. NEON data are provided under NEON's open data policy.
