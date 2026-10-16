# CRS Points Calculator

Estimate your Express Entry CRS score under today's rules and under the rules that start November 3, 2026, when Canadian work in high-wage occupations and job offers earn new points.

**Unofficial estimate. Not affiliated with IRCC or the Government of Canada.**

**Live site:** https://kmehta2228.github.io/crs-points-calculator/

## What's here

| Path | What it does |
| --- | --- |
| `site/index.html` | The whole calculator in one file. No build step, no server. Google Analytics counts visits, with a cookie banner; answers stay in each visitor's own browser. |
| `site/data/draws.json` | The data behind the next-draw estimates: recent IRCC draws, the pool, and estimated cut-offs by draw type and draw size. Rebuilt by the update workflow; don't edit it by hand. |
| `.github/workflows/deploy-pages.yml` | Publishes `site/` to GitHub Pages every time `site/` changes on `main`, and when the update workflow asks it to. |
| `.github/workflows/update-draws.yml` | A few times a day, refreshes `site/data/draws.json` from IRCC's draw data, commits it if it changed and redeploys the site. |
| `.github/workflows/watch-ircc.yml` | Every Monday, checks the IRCC pages the calculator is built from and opens an issue if any of them changed. |
| `.github/scripts/update_draws.py` | What the update workflow runs: downloads IRCC's rounds file, checks it, then runs the builder. |
| `.github/scripts/build_predictions.py` | Builds `draws.json` from IRCC's rounds file. |
| `.github/scripts/crs_engine/` | The three prediction models and the harness they were tested with. |
| `.github/scripts/check_ircc.py` | The check the watch workflow runs. Keeps a fingerprint of each page in `.github/ircc-snapshot.json`. |

## Updating the calculator

Edit `site/index.html` and push to `main`. The live site updates in about a minute.

- CRS point values are in the `SCORING` section of the script.
- The high-wage occupations list is in `NOC_RAW`, one `tier|NOC code|title` line per occupation.
- New visitors start from the empty answers in `BLANK`. The sample answers behind "Show an example" are in `EXAMPLE`.
- The Google Analytics measurement ID is in the `gtag` snippet at the top of `<head>`. The same snippet lists the regions where analytics cookies stay off until the visitor clicks Allow (EU/EEA, UK, Switzerland, Quebec); everywhere else they're on until the visitor clicks Decline. The banner itself is at the end of `<body>`.

The first run of the watch workflow saves a baseline and opens no issue. GitHub pauses scheduled workflows in public repositories after 60 days without any repository activity; if that happens, re-enable them from the Actions tab.

## Next draw estimates (for maintainers)

For each draw type (CEC, French, healthcare, PNP and so on) and each number of invitations, the site estimates the lowest score that would get an invitation in the next draw, with a range that held the real cut-off about 8 times in 10 when tested on past draws. Each estimate is the median of three models: a queue model per draw type, a model of IRCC's published pool score distribution, and a regression on past cut-offs. Estimates for draws after November 3, 2026 add a scenario for the new points. They are estimates, not IRCC numbers.

**Data.** Everything comes from IRCC's [rounds of invitations file](https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json), the one behind its [rounds page](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/rounds-invitations.html). Statistics Canada does not publish Express Entry draw data.

**Schedule.** `update-draws.yml` runs on weekdays at 15:17 and 17:47 UTC and every day at 20:47 UTC: 11:17 a.m., 1:47 p.m. and 4:47 p.m. in Toronto in summer (EDT), an hour earlier in winter (EST). New draws usually reach the site 1 to 3 hours after IRCC posts them. A run that finds nothing new changes nothing. If the download or the data looks wrong (an incomplete file, a draw dated in the future or out of order, fewer draws than before, or score bands IRCC has re-cut on its rounds page), the run fails, the site keeps its last good data, and three failed scheduled runs in a row open an issue. The run log's error line says what was wrong.

**Run it locally** (Python 3.9 or newer, standard library only, about 10 seconds):

```sh
python3 -I .github/scripts/update_draws.py                       # download IRCC's file, update site/data/draws.json
python3 -I .github/scripts/update_draws.py --input rounds.json   # use a saved copy of IRCC's file instead
```

The file only changes when IRCC's data or the code does. After changing `build_predictions.py`, run it and commit `site/data/draws.json` with your change, or run the workflow from the Actions tab after you push.

**Yearly maintenance**

- When the 2027-2029 Immigration Levels Plan is published (due fall 2026), and each January, update `QUOTA_FACTS` in `build_predictions.py` and bump `MODEL_VERSION`.
- The files in `crs_engine/` must stay byte-identical to the versions that were tested; their SHA-256 hashes are published in `draws.json`. Change them only after re-running the backtest with `python3 -I .github/scripts/crs_engine/harness.py .github/scripts/crs_engine/approach_*.py --data ee_rounds_123_en.json --no-write`.
- Review the first draw cycle after November 3, 2026 by hand. The after-November-3 shifts (`NOV3_DELTA` in `build_predictions.py`) are a reasoned scenario, not fitted to data.
- If IRCC changes its rounds page or score bands, update `BAND_LABELS` in `update_draws.py` and the models (re-tested) before publishing again.

## Sources

- [CRS criteria](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/crs-criteria.html)
- [High-wage occupations list](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/high-wage.html)
- [2026 changes to points for Canadian work experience and job offers](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/changes.html)
- [Express Entry language test results](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/documents/language-test.html)
- [Express Entry rounds of invitations](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/rounds-invitations.html)
