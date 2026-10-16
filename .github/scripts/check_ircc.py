#!/usr/bin/env python3
"""Weekly check of the IRCC pages the calculator is built from.

Fetches each page, keeps only its main content (dropping the "Date modified"
footer, which changes on every republish), and compares a fingerprint with the
last saved snapshot. Writes changed=true/false to $GITHUB_OUTPUT, and an issue
body to ircc-changes.md when something changed.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser

PAGES = {
    "CRS criteria": "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/crs-criteria.html",
    "High-wage occupations list": "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/check-score/high-wage.html",
    "2026 changes to CRS points": "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/changes.html",
    "Express Entry language test results": "https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry/documents/language-test.html",
}
SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ircc-snapshot.json")
ISSUE_BODY = "ircc-changes.md"
SKIPPED_TAGS = ("script", "style", "noscript", "template")


class MainText(HTMLParser):
    """Collects the visible text inside <main>, skipping scripts and styles."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_main = 0
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.in_main += 1
        elif tag in SKIPPED_TAGS:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "main" and self.in_main:
            self.in_main -= 1
        elif tag in SKIPPED_TAGS and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if self.in_main and not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


def main_text(html):
    parser = MainText()
    parser.feed(html)
    text = "\n".join(parser.parts)
    # The page-details footer holds the "Date modified" stamp; drop it and everything after it.
    text = re.split(r"\n(?:Page details|Date modified)\b", text)[0]
    if len(text) < 200:
        raise RuntimeError("the page content looks empty, so its layout may have changed")
    return text


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CRS calculator page watch; GitHub Actions)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def main(fetch_page=fetch):
    try:
        with open(SNAPSHOT, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}

    new, changed, failed = {}, [], []
    for name, url in PAGES.items():
        try:
            digest = hashlib.sha256(main_text(fetch_page(url)).encode("utf-8")).hexdigest()
        except Exception as exc:  # keep the old fingerprint so a failed fetch never looks like a change
            failed.append(f"{name}: {exc}")
            if name in old:
                new[name] = old[name]
            continue
        new[name] = {"url": url, "sha256": digest}
        if name in old and old[name].get("sha256") != digest:
            changed.append((name, url))

    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2)
        f.write("\n")

    if changed:
        lines = ["IRCC changed one or more pages the calculator is built from:", ""]
        lines += [f"- [{name}]({url})" for name, url in changed]
        lines += [
            "",
            "Review what changed and update `site/index.html` if needed. Point values are in the "
            "`SCORING` section of the script, and the high-wage occupations are in `NOC_RAW`.",
        ]
        with open(ISSUE_BODY, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")

    for message in failed:
        print(f"::warning::Could not check {message}")
    print(f"Checked {len(PAGES)} pages: {len(changed)} changed, {len(failed)} could not be fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
