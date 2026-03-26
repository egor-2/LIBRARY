#!/usr/bin/env python3
"""Validate records.csv: header, field constraints, and duplicate detection."""

import csv
import os
import re
import sys

EXPECTED_HEADER = ["FOLDER", "ARXIV_ID", "OPENREVIEW_ID", "TITLE", "AUTHORS", "KEYWORDS", "COMMENT"]

RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records.csv")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalised_distance(a: str, b: str) -> float:
    maxlen = max(len(a), len(b))
    if maxlen == 0:
        return 0.0
    return levenshtein(a, b) / maxlen


def check_title(title: str) -> list[str]:
    errors = []
    if not title:
        errors.append("TITLE is empty")
        return errors
    if title != title.lower():
        errors.append(f"TITLE contains uppercase characters: '{title}'")
    if re.search(r"[^a-z ]", title):
        errors.append(f"TITLE contains characters other than lowercase letters and spaces: '{title}'")
    if "  " in title:
        errors.append(f"TITLE contains consecutive spaces: '{title}'")
    if title != title.strip():
        errors.append(f"TITLE has leading/trailing whitespace")
    return errors


def check_authors(authors: str) -> list[str]:
    errors = []
    if not authors:
        errors.append("AUTHORS is empty")
        return errors
    for part in authors.split(","):
        name = part.strip()
        if not name:
            errors.append(f"AUTHORS has empty name (double comma or trailing comma): '{authors}'")
            continue
        if re.search(r"[^a-z ]", name):
            errors.append(f"AUTHORS contains invalid characters in '{name}' — only lowercase letters and spaces allowed")
        if "  " in name:
            errors.append(f"AUTHORS has consecutive spaces in '{name}'")
    return errors


def check_keywords(keywords: str) -> list[str]:
    errors = []
    if not keywords:
        errors.append("KEYWORDS is empty")
        return errors
    kw_list = [k.strip() for k in keywords.split(",")]
    kw_list = [k for k in kw_list if k]
    if len(kw_list) != 20:
        errors.append(f"KEYWORDS should have exactly 20 keywords, found {len(kw_list)}")
    for kw in kw_list:
        if re.search(r"[^a-z0-9 ]", kw):
            errors.append(f"KEYWORDS contains invalid characters in '{kw}'")
    return errors


def check_arxiv_id(arxiv_id: str) -> list[str]:
    if not arxiv_id:
        return []  # can be empty
    if not re.fullmatch(r"\d+\.\d+", arxiv_id):
        return [f"ARXIV_ID has wrong format '{arxiv_id}' — expected digits.digits (e.g. 1234.56789)"]
    return []


def check_folder(folder: str) -> list[str]:
    if not folder:
        return ["FOLDER is empty"]
    if not re.fullmatch(r"[a-z0-9_-]+", folder):
        return [f"FOLDER '{folder}' should only contain lowercase letters, digits, hyphens, underscores"]
    return []


def main() -> int:
    if not os.path.exists(RECORDS_PATH):
        print("ERROR: records.csv not found")
        return 1

    with open(RECORDS_PATH, newline="", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        print("WARNING: records.csv is empty (no header, no records)")
        return 0

    lines = content.strip().split("\n")
    reader = csv.reader(lines)
    rows = list(reader)

    if not rows:
        print("ERROR: records.csv has no rows")
        return 1

    header = rows[0]
    errors: list[str] = []

    if header != EXPECTED_HEADER:
        errors.append(f"Header mismatch.\n  Expected: {EXPECTED_HEADER}\n  Got:      {header}")
        # try to continue anyway if column count matches
        if len(header) != len(EXPECTED_HEADER):
            for e in errors:
                print(f"ERROR: {e}")
            return 1

    records = rows[1:]
    if not records:
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            return 1
        print("OK: header valid, no records yet")
        return 0

    # Validate each row
    seen_folders: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_openreview: dict[str, int] = {}
    titles: list[tuple[int, str]] = []

    for i, row in enumerate(records, start=2):  # line numbers (1-indexed, header=1)
        if len(row) != len(EXPECTED_HEADER):
            errors.append(f"Row {i}: expected {len(EXPECTED_HEADER)} fields, got {len(row)}")
            continue

        folder, arxiv_id, openreview_id, title, authors, keywords, comment = row

        for e in check_folder(folder):
            errors.append(f"Row {i} ({folder}): {e}")
        for e in check_arxiv_id(arxiv_id):
            errors.append(f"Row {i} ({folder}): {e}")
        for e in check_title(title):
            errors.append(f"Row {i} ({folder}): {e}")
        for e in check_authors(authors):
            errors.append(f"Row {i} ({folder}): {e}")
        for e in check_keywords(keywords):
            errors.append(f"Row {i} ({folder}): {e}")

        # Duplicate: FOLDER
        if folder:
            if folder in seen_folders:
                errors.append(f"Row {i}: duplicate FOLDER '{folder}' (first seen row {seen_folders[folder]})")
            else:
                seen_folders[folder] = i

        # Duplicate: ARXIV_ID
        if arxiv_id:
            if arxiv_id in seen_arxiv:
                errors.append(f"Row {i} ({folder}): duplicate ARXIV_ID '{arxiv_id}' (first seen row {seen_arxiv[arxiv_id]})")
            else:
                seen_arxiv[arxiv_id] = i

        # Duplicate: OPENREVIEW_ID
        if openreview_id:
            if openreview_id in seen_openreview:
                errors.append(f"Row {i} ({folder}): duplicate OPENREVIEW_ID '{openreview_id}' (first seen row {seen_openreview[openreview_id]})")
            else:
                seen_openreview[openreview_id] = i

        if title:
            titles.append((i, title))

    # Suspicious title similarity (Levenshtein)
    SIMILARITY_THRESHOLD = 0.15  # titles within 15% edit distance are flagged
    for a in range(len(titles)):
        for b in range(a + 1, len(titles)):
            row_a, title_a = titles[a]
            row_b, title_b = titles[b]
            dist = normalised_distance(title_a, title_b)
            if dist <= SIMILARITY_THRESHOLD:
                errors.append(
                    f"Suspiciously similar titles (distance {dist:.2f}):\n"
                    f"  Row {row_a}: '{title_a}'\n"
                    f"  Row {row_b}: '{title_b}'"
                )

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print(f"OK: {len(records)} record(s) validated, no issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
