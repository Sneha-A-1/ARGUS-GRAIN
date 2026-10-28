# Data

The evaluation runs on two open NEON data products from the **Konza Prairie
Agricultural Site (KONA)**, Kansas, 1 May - 31 August 2023:

| Product | Contents | File |
|---|---|---|
| DP1.00094.001 | soil water content, soil ion content, quality flags | `raw/SWS_30_minute.csv` |
| DP1.00041.001 | soil temperature | `raw/ST_30_minute.csv` |

Both files are included in this repository (21 MB total), so `run_all.py`
works with no download.

## Re-downloading

```bash
pip install neonutilities
```

```python
import neonutilities as nu
TOKEN = "your_neon_api_token"      # required since June 2026, free from neonscience.org

for dpid in ["DP1.00094.001", "DP1.00041.001"]:
    nu.zips_by_product(dpid=dpid, site="KONA",
                       startdate="2023-05", enddate="2023-08",
                       package="basic", savepath="./neon",
                       check_size=False, token=TOKEN)

nu.stack_by_table("./neon/filesToStack00094", savepath="./neon")
nu.stack_by_table("./neon/filesToStack00041", savepath="./neon")
```

Keep these columns and place the results in `raw/`:

* SWS: `startDateTime, horizontalPosition, verticalPosition, VSWCMean, VSWCFinalQF, VSICMean`
* ST: `startDateTime, horizontalPosition, verticalPosition, soilTempMean`

## Deployment model

`horizontalPosition` is the soil plot (1-5) and `verticalPosition` the
measurement depth (501-508). `src/data.py` treats each plot as a site and each
(plot, depth) pair as a sensor node with three streams: volumetric water
content, soil ion content and soil temperature.

## Data quality

The records are used as released. Roughly 15 % of raw rows are empty and only
60.3 % pass the NEON quality flag. Neither is repaired: gaps and flagged
samples are part of what a retrieval framework has to survive, and the
reported reconstruction errors therefore include some instrument noise.

## Using a different site

Any NEON site works - substitute the site code. `STER` (Sterling, Colorado) is
the other agricultural site. The agronomic bounds in `data.TH` are set from
KONA's measured distributions and should be re-derived for a different soil.
