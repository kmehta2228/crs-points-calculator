#!/usr/bin/env python3
"""Refresh site/data/draws.json, the data behind the site's next-draw estimates.

Downloads IRCC's Express Entry rounds file (ee_rounds_123_en.json), makes sure the download is
complete, checks that IRCC's rounds page still labels the pool-distribution fields dd1..dd17
with the score bands the model expects, then runs build_predictions.py on it with --previous
set to the published file. The builder rewrites site/data/draws.json only when its content
changes, and writes changed=true|false and latest=<draw number> to $GITHUB_OUTPUT.

Exits 1 and writes nothing when the data can't be trusted: the download failed or came back
partial, empty or as an HTML page; a draw is dated in the future or out of order; IRCC re-cut
the score bands; or the builder refused the file (for example fewer rounds, or an older newest
draw, than the published file). The site then keeps its last good data.

  python3 -I .github/scripts/update_draws.py                      # download from canada.ca
  python3 -I .github/scripts/update_draws.py --input ROUNDS.json  # offline, from a saved copy
"""
import argparse
import datetime as dt
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from html.parser import HTMLParser

sys.dont_write_bytecode = True  # never leave __pycache__ folders in the repository

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "build_predictions.py")
DRAWS = os.path.normpath(os.path.join(HERE, "..", "..", "site", "data", "draws.json"))

ROUNDS_URL = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
ROUNDS_PAGE = "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/rounds-invitations.html"

# canada.ca's CDN silently stalls many user agents. The first is the one check_ircc.py already
# uses from GitHub Actions; None falls back to urllib's own "Python-urllib/3.x". Test any change.
USER_AGENTS = ("Mozilla/5.0 (CRS calculator page watch; GitHub Actions)", None)
TIMEOUT = 60  # seconds for the connection and for each read
ATTEMPT_SECONDS = 120  # a download attempt that takes longer than this is abandoned
WAITS = (10, 30, 60)  # seconds to wait before the 2nd, 3rd and 4th attempt
MIN_BYTES, MAX_BYTES = 100_000, 20_000_000  # the same limits build_predictions.py enforces

# The score range IRCC's rounds page shows next to each distribution field. The model
# (crs_engine/harness.py, BANDS) reads dd1..dd17 as exactly these ranges.
BAND_LABELS = {
    "dd1": "601-1200", "dd2": "501-600", "dd3": "451-500", "dd4": "491-500",
    "dd5": "481-490", "dd6": "471-480", "dd7": "461-470", "dd8": "451-460",
    "dd9": "401-450", "dd10": "441-450", "dd11": "431-440", "dd12": "421-430",
    "dd13": "411-420", "dd14": "401-410", "dd15": "351-400", "dd16": "301-350",
    "dd17": "0-300",
}
FIELD = re.compile(r"#\[roundsofinvitation\]/rounds/0/(dd\d+)")

# Every round in IRCC's file has had exactly these fields since 2015.
ROUND_FIELDS = {
    "drawNumber", "drawNumberURL", "drawDate", "drawDateFull", "drawName", "drawSize",
    "drawCRS", "mitext", "DrawText1", "drawText2", "drawDateTime", "drawCutOff",
    "drawDistributionAsOn",
} | {f"dd{i}" for i in range(1, 19)}

# Anything that can go wrong in one download attempt (HTTP errors and timeouts are OSErrors;
# a connection cut short raises http.client.IncompleteRead; bad bodies raise ValueError).
DOWNLOAD_ERRORS = (OSError, http.client.HTTPException, ValueError, EOFError, zlib.error)


class Refuse(Exception):
    """The data can't be trusted: publish nothing and keep the last good file."""


def gunzip(data):
    """Decompress a whole gzip body. Its CRC and length trailer catch a cut-off download."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    body = inflater.decompress(data, MAX_BYTES + 1)
    if len(body) > MAX_BYTES or inflater.unconsumed_tail:
        raise ValueError(f"more than {MAX_BYTES:,} bytes after decompression")
    if not inflater.eof or inflater.unused_data:
        raise ValueError("the gzip body is cut off or has extra bytes after it")
    return body


def download(url, agent, accept):
    """One GET of url. Returns (body, headers); raises on any sign of a partial response."""
    headers = {"Accept": accept, "Accept-Encoding": "gzip"}
    if agent:
        headers["User-Agent"] = agent
    deadline = time.monotonic() + ATTEMPT_SECONDS
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        chunks, size = [], 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError(f"the response is larger than {MAX_BYTES:,} bytes")
            if time.monotonic() > deadline:
                raise ValueError(f"the download took more than {ATTEMPT_SECONDS} seconds")
            chunks.append(chunk)
        body = b"".join(chunks)
        # read() returns early instead of failing when the connection drops, so compare
        # what arrived with what the server announced.
        announced = response.headers.get("Content-Length")
        if announced is not None and int(announced) != len(body):
            raise ValueError(f"the connection closed after {len(body):,} of {int(announced):,} bytes")
        encoding = (response.headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding == "gzip":
            body = gunzip(body)
        elif encoding != "identity":
            raise ValueError(f"unexpected Content-Encoding {encoding!r}")
        return body, response.headers


def fetch(url, accept, attempts, check):
    """Download url until check(body, headers) accepts it, waiting longer between attempts and
    alternating user agents. Returns (check's result, headers); raises Refuse after the last
    failed attempt."""
    problems = []
    for attempt in range(attempts):
        if attempt:
            time.sleep(WAITS[min(attempt, len(WAITS)) - 1])
        try:
            body, headers = download(url, USER_AGENTS[attempt % len(USER_AGENTS)], accept)
            return check(body, headers), headers
        except DOWNLOAD_ERRORS as exc:
            problems.append(f"attempt {attempt + 1}: {exc}")
            print(f"Download attempt {attempt + 1} of {url} failed: {exc}")
    raise Refuse(f"could not download {url} ({'; '.join(problems)})")


def parse_rounds_file(body, headers=None):
    """(clean bytes, parsed JSON) of an IRCC rounds file; ValueError for an empty, HTML,
    cut-off or otherwise wrong body."""
    if headers is not None and "html" in (headers.get("Content-Type") or "").lower():
        raise ValueError(f"the server sent an HTML page ({headers.get('Content-Type')})")
    if body.startswith(b"\xef\xbb\xbf"):
        print("::warning::IRCC's rounds file now starts with a byte-order mark; it was removed "
              "before building, so provenance.input_sha256 is of the file without it")
        body = body[3:]
    if len(body) < MIN_BYTES:
        raise ValueError(f"the body is only {len(body):,} bytes")
    text = body.decode("utf-8")
    if not text.lstrip().startswith("{"):
        raise ValueError(f"the body is not JSON (it starts with {text.lstrip()[:40]!r})")
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("rounds"), list):
        raise ValueError("the JSON has no 'rounds' list")
    return body, data


def check_dates(data, today):
    """Refuse a drawDate that isn't YYYY-MM-DD, lies in the future, or goes backwards as draw
    numbers go up: one typo would move every expected draw date on the site."""
    last_ok = (today + dt.timedelta(days=1)).isoformat()
    dated = []
    for r in data["rounds"]:
        if not isinstance(r, dict):
            continue  # build_predictions.py refuses these
        number = str(r.get("drawNumber", "")).strip()
        date = str(r.get("drawDate", "")).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise Refuse(f"round {number!r} has drawDate {r.get('drawDate')!r}, not YYYY-MM-DD")
        if date > last_ok:
            raise Refuse(f"round {number!r} is dated {date}, after today ({today})")
        match = re.fullmatch(r"(\d+)([a-z]?)", number)
        if match:  # build_predictions.py refuses other numbers
            dated.append((int(match.group(1)), match.group(2), date, number))
    dated.sort()
    for earlier, later in zip(dated, dated[1:]):
        if later[2] < earlier[2]:
            raise Refuse(f"round {later[3]!r} is dated {later[2]}, before round {earlier[3]!r} ({earlier[2]})")


def field_warnings(data):
    """A warning when IRCC adds or drops fields (the model only needs the ones it reads)."""
    newest = next((r for r in data["rounds"] if isinstance(r, dict)), {})
    added, missing = sorted(set(newest) - ROUND_FIELDS), sorted(ROUND_FIELDS - set(newest))
    if added or missing:
        return [f"IRCC changed the fields of its rounds file: added {added}, missing {missing}"]
    return []


class RoundsPage(HTMLParser):
    """Reads IRCC's rounds page template: the JSON files its data-wb-jsonmanager blocks load,
    and the first-cell text of every table row that shows a distribution field
    (data-json-replace="#[roundsofinvitation]/rounds/0/ddN")."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []
        self.labels = {}  # "dd2" -> ["501-600"], one entry per row showing dd2
        self.cells = None  # cell texts of the current table row
        self.fields = []  # ddN fields shown in the current table row
        self.text = None  # text pieces of the current cell

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.end_row()
            self.cells = []
        elif tag in ("td", "th") and self.cells is not None:
            self.end_cell()
            self.text = []
        for name, value in attrs:
            if name == "data-wb-jsonmanager" and value:
                self.sources.append(value)
            elif name == "data-json-replace" and value and self.cells is not None:
                match = FIELD.fullmatch(value.strip())
                if match:
                    self.fields.append(match.group(1))

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.end_cell()
        elif tag in ("tr", "thead", "tbody", "tfoot", "table"):
            self.end_row()

    def handle_data(self, data):
        if self.text is not None:
            self.text.append(data)

    def end_cell(self):
        if self.text is not None and self.cells is not None:
            self.cells.append(" ".join("".join(self.text).split()))
        self.text = None

    def end_row(self):
        self.end_cell()
        if self.cells:
            for field in self.fields:
                self.labels.setdefault(field, []).append(self.cells[0])
        self.cells, self.fields = None, []


def band_label(text):
    """Score-range label with plain hyphens and no spaces around them ('601 - 1200' -> '601-1200')."""
    text = re.sub("[\u2010-\u2015\u2212]", "-", text)  # hyphens, en and em dashes, minus
    return re.sub(r"\s*-\s*", "-", " ".join(text.split()))


def check_template(html):
    """Refuse when IRCC's rounds page no longer loads ee_rounds_123_en.json as
    "roundsofinvitation" or labels any of dd1..dd17 with another score range (re-cut bands)."""
    page = RoundsPage()
    page.feed(html)
    page.close()
    if not any("roundsofinvitation" in s and "/ee_rounds_123_en.json" in s for s in page.sources):
        raise Refuse("IRCC's rounds page no longer loads ee_rounds_123_en.json as "
                     f"'roundsofinvitation' (it loads {page.sources or 'nothing'}); check {ROUNDS_PAGE}")
    wrong = []
    for field, want in BAND_LABELS.items():
        found = [band_label(label) for label in page.labels.get(field, [])]
        if not found or any(label != want for label in found):
            wrong.append(f"{field} is {', '.join(map(repr, found)) or 'not shown'} instead of '{want}'")
    if wrong:
        raise Refuse("IRCC's rounds page shows different score bands: " + "; ".join(wrong)
                     + f". See {ROUNDS_PAGE}. The model reads dd1..dd17 as the old bands, so "
                     "crs_engine and BAND_LABELS need updating (and re-validating) first.")


def check_rounds_page(page_file):
    """Run check_template on the live page, or on a saved copy. A page that can't be downloaded
    only earns a warning: it is a cross-check, and the rounds file itself passed every test."""
    if page_file:
        try:
            with open(page_file, "rb") as f:
                html = f.read().decode("utf-8", "replace")
        except OSError as exc:
            raise Refuse(f"{page_file}: {exc}") from exc
    else:
        try:
            html, _ = fetch(ROUNDS_PAGE, "text/html", 2,
                            lambda body, headers: body.decode("utf-8", "replace"))
        except Refuse as exc:
            print(f"::warning::Skipped the band-label check: {exc}")
            return
    check_template(html)
    print("IRCC's rounds page still labels dd1..dd17 with the expected score bands.")


def build(raw_path, out):
    """Run build_predictions.py; returns its exit code (0 = written or unchanged, 1 = refused).
    It inherits $GITHUB_OUTPUT and writes changed= and latest= there."""
    command = [sys.executable, "-I", "-B", BUILDER, raw_path, out, "--previous", out]
    sys.stdout.flush()  # keep the log in order: the builder writes to the same stdout
    return subprocess.run(command, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh site/data/draws.json from IRCC's rounds file.")
    parser.add_argument("--input", metavar="FILE", help="use this saved copy of ee_rounds_123_en.json "
                        "instead of downloading it (also skips the rounds-page check unless --page is given)")
    parser.add_argument("--page", metavar="FILE", help="check this saved copy of the rounds page "
                        "instead of downloading it")
    parser.add_argument("--out", default=DRAWS, help="file to update (default: site/data/draws.json)")
    parser.add_argument("--today", type=dt.date.fromisoformat, metavar="YYYY-MM-DD",
                        help="the date to check draw dates against (default: today in UTC; for tests)")
    args = parser.parse_args(argv)
    today = args.today or dt.datetime.now(dt.timezone.utc).date()

    workdir = tempfile.mkdtemp(prefix="ircc-rounds-")
    try:
        if args.input:
            try:
                with open(args.input, "rb") as f:
                    body, data = parse_rounds_file(f.read())
            except (OSError, ValueError) as exc:
                raise Refuse(f"{args.input}: {exc}") from exc
            print(f"Read {args.input}: {len(body):,} bytes.")
        else:
            (body, data), headers = fetch(ROUNDS_URL, "application/json", 4, parse_rounds_file)
            print(f"Downloaded {ROUNDS_URL}: {len(body):,} bytes, "
                  f"last modified {headers.get('Last-Modified', 'unknown')}.")
        check_dates(data, today)
        for warning in field_warnings(data):
            print(f"::warning::{warning}")
        if args.page or not args.input:
            check_rounds_page(args.page)
        else:
            print("Skipped the rounds-page check (offline run; pass --page FILE to check a saved copy).")
        raw_path = os.path.join(workdir, "ee_rounds_123_en.json")
        with open(raw_path, "wb") as f:
            f.write(body)
        return 0 if build(raw_path, args.out) == 0 else 1
    except Refuse as exc:
        print(f"::error::Refusing to publish draw data: {exc}")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
