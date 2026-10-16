# CRS Points Calculator

Estimate your Express Entry CRS score under today's rules and under the rules that start November 3, 2026, when Canadian work in high-wage occupations and job offers earn new points.

**Unofficial estimate. Not affiliated with IRCC or the Government of Canada.**

**Live site:** https://kmehta2228.github.io/crs-points-calculator/

## What's here

| Path | What it does |
| --- | --- |
| `site/index.html` | The whole calculator in one file. No build step, no server, no tracking. Answers stay in each visitor's own browser. |
| `.github/workflows/deploy-pages.yml` | Publishes `site/` to GitHub Pages every time `site/` changes on `main`. |
| `.github/workflows/watch-ircc.yml` | Every Monday, checks the IRCC pages the calculator is built from and opens an issue if any of them changed. |
| `.github/scripts/check_ircc.py` | The check the watch workflow runs. Keeps a fingerprint of each page in `.github/ircc-snapshot.json`. |

## Updating the calculator

Edit `site/index.html` and push to `main`. The live site updates in about a minute.

- CRS point values are in the `SCORING` section of the script.
- The high-wage occupations list is in `NOC_RAW`, one `tier|NOC code|title` line per occupation.

The first run of the watch workflow saves a baseline and opens no issue. GitHub pauses scheduled workflows in public repositories after 60 days without any repository activity; if that happens, re-enable it from the Actions tab.

## Sources

- [CRS criteria](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/crs-criteria.html)
- [High-wage occupations list](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/high-wage.html)
- [2026 changes to points for Canadian work experience and job offers](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/changes.html)
- [Express Entry language test results](https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/documents/language-test.html)
