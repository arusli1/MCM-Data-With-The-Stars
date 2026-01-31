"""
Build a combined Wikipedia edit-count dataset with:
  - preseason edits (already computed in Data/dwts_wiki_edits_preseason.csv)
  - end-of-season edits (computed here)
  - change = end - start

Inputs:
  - Archit_Preliminary/celebrity_season_premiere_table.csv
      columns: season, season_premiere_date, season_finale_date, celebrity_name
  - Data/dwts_wiki_edits_preseason.csv
      columns: season, celebrityname, wiki_title, pageid, cutoff_utc, edits_total_to_cutoff, status, notes

Output:
  - Data/dwts_wiki_edits_start_end_change.csv (incremental append/resumable)

End-of-season cutoff rule:
  For season finale date D (YYYY-MM-DD):
    cutoff_end_utc = (D + 1 day) at 00:00:00Z
  This includes all edits made on the finale date itself (up to 23:59:59Z).

Wikipedia API:
  https://en.wikipedia.org/w/api.php
  action=query&prop=revisions&pageids={pageid}&rvprop=ids|timestamp&rvlimit=max&rvdir=older&rvstart={cutoff_end_utc}
  Use MediaWiki continuation (rvcontinue) until exhausted.

Dependencies:
  requests, pandas, datetime, time
"""

from __future__ import annotations

import csv
import datetime as dt
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"


def _sleep_polite(seconds: float = 0.15) -> None:
    time.sleep(seconds)


def _requests_get_json(session: requests.Session, params: Dict, timeout: float = 20.0) -> Dict:
    _sleep_polite()
    resp = session.get(WIKI_API, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def cutoff_end_utc(finale_date_iso: str) -> str:
    """
    finale_date_iso: YYYY-MM-DD
    returns: (finale + 1 day) at 00:00:00Z
    """
    d = dt.date.fromisoformat(finale_date_iso)
    d2 = d + dt.timedelta(days=1)
    cutoff_dt = dt.datetime.combine(d2, dt.time(0, 0, 0), tzinfo=dt.timezone.utc)
    return cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def count_revisions_to_cutoff(session: requests.Session, pageid: int, cutoff_iso: str) -> Tuple[Optional[int], str, str]:
    """
    Count revisions up to cutoff_iso (inclusive) for one pageid, using continuation.
    Returns: (count, status, notes)
    """
    count = 0
    cont: Dict[str, str] = {}

    while True:
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "pageids": str(pageid),
            "rvprop": "ids|timestamp",
            "rvlimit": "max",
            "rvdir": "older",
            "rvstart": cutoff_iso,
        }
        params.update(cont)

        try:
            data = _requests_get_json(session, params)
        except Exception as exc:
            return None, "api_error", f"count_revisions request failed: {exc}"

        pages = data.get("query", {}).get("pages", {})
        page = pages.get(str(pageid), {})
        revs = page.get("revisions", []) or []
        count += len(revs)

        cont_data = data.get("continue")
        if not cont_data:
            break
        cont = {k: str(v) for k, v in cont_data.items()}

    return count, "ok", ""


def _detect_col(df: pd.DataFrame, candidates) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_existing_keys(out_csv: str) -> set[tuple[int, str]]:
    try:
        prev = pd.read_csv(out_csv)
    except FileNotFoundError:
        return set()
    if prev.empty:
        return set()
    season_col = _detect_col(prev, ["season"])
    name_col = _detect_col(prev, ["celebrityname", "celebrity_name", "celebrity"])
    if not season_col or not name_col:
        return set()
    keys = set()
    for _, r in prev.iterrows():
        try:
            s = int(r[season_col])
        except Exception:
            continue
        n = str(r[name_col]).strip()
        if n:
            keys.add((s, n))
    return keys


def _append_row(out_csv: str, row: Dict) -> None:
    fieldnames = [
        "season",
        "celebrityname",
        "wiki_title",
        "pageid",
        "cutoff_start_utc",
        "edits_start_to_cutoff",
        "cutoff_end_utc",
        "edits_end_to_cutoff",
        "edits_change",
        "status_start",
        "status_end",
        "notes",
    ]

    write_header = False
    try:
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            _ = f.readline()
    except FileNotFoundError:
        write_header = True

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    season_table_csv = "Archit_Preliminary/celebrity_season_premiere_table.csv"
    preseason_csv = "Data/dwts_wiki_edits_preseason.csv"
    out_csv = "Data/dwts_wiki_edits_start_end_change.csv"

    table = pd.read_csv(season_table_csv)
    season_col = _detect_col(table, ["season"])
    name_col = _detect_col(table, ["celebrity_name", "celebrityname", "celebrity"])
    finale_col = _detect_col(table, ["season_finale_date"])
    if not season_col or not name_col or not finale_col:
        raise ValueError(f"{season_table_csv} must have columns: season, celebrity_name, season_finale_date")

    table = table.rename(columns={season_col: "season", name_col: "celebrityname", finale_col: "season_finale_date"})
    table["season"] = pd.to_numeric(table["season"], errors="raise").astype(int)
    table["celebrityname"] = table["celebrityname"].astype(str).str.strip()

    pre = pd.read_csv(preseason_csv)
    pre_season_col = _detect_col(pre, ["season"])
    pre_name_col = _detect_col(pre, ["celebrityname", "celebrity_name", "celebrity"])
    if not pre_season_col or not pre_name_col:
        raise ValueError(f"{preseason_csv} must have season and celebrityname columns")

    pre = pre.rename(columns={pre_season_col: "season", pre_name_col: "celebrityname"})
    pre["season"] = pd.to_numeric(pre["season"], errors="coerce").astype("Int64")
    pre["celebrityname"] = pre["celebrityname"].astype(str).str.strip()

    merged = table.merge(pre, on=["season", "celebrityname"], how="left", suffixes=("", "_pre"))

    done = _load_existing_keys(out_csv)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "DWTS-MCM-2026 Wikipedia revisions counter (educational) - contact: your_email@example.com"}
    )

    for r in merged.itertuples(index=False):
        season = int(r.season)
        name = str(r.celebrityname)
        if (season, name) in done:
            continue

        # Pull start-of-season fields from preseason CSV
        wiki_title = getattr(r, "wiki_title", "")
        pageid = getattr(r, "pageid", "")
        cutoff_start = getattr(r, "cutoff_utc", "")
        edits_start = getattr(r, "edits_total_to_cutoff", "")
        status_start = getattr(r, "status", "")
        notes_start = getattr(r, "notes", "")

        finale_date = getattr(r, "season_finale_date", "")
        if not isinstance(finale_date, str):
            finale_date = "" if pd.isna(finale_date) else str(finale_date)
        finale_date = finale_date.strip()

        if not finale_date or finale_date.lower() == "nan":
            _append_row(
                out_csv,
                {
                    "season": season,
                    "celebrityname": name,
                    "wiki_title": wiki_title,
                    "pageid": pageid,
                    "cutoff_start_utc": cutoff_start,
                    "edits_start_to_cutoff": edits_start,
                    "cutoff_end_utc": "",
                    "edits_end_to_cutoff": "",
                    "edits_change": "",
                    "status_start": status_start,
                    "status_end": "missing_finale_date",
                    "notes": f"Missing season_finale_date for season {season}. {notes_start}".strip(),
                },
            )
            done.add((season, name))
            continue

        cutoff_end = cutoff_end_utc(finale_date)

        # If we don't have a valid pageid from preseason, we cannot count end edits reliably.
        try:
            pageid_int = int(pageid)
        except Exception:
            _append_row(
                out_csv,
                {
                    "season": season,
                    "celebrityname": name,
                    "wiki_title": wiki_title,
                    "pageid": pageid,
                    "cutoff_start_utc": cutoff_start,
                    "edits_start_to_cutoff": edits_start,
                    "cutoff_end_utc": cutoff_end,
                    "edits_end_to_cutoff": "",
                    "edits_change": "",
                    "status_start": status_start,
                    "status_end": "missing_pageid",
                    "notes": f"No valid pageid from preseason CSV; cannot compute end edits. {notes_start}".strip(),
                },
            )
            done.add((season, name))
            continue

        # Count end edits (even if start status wasn't ok, we still can attempt if pageid exists)
        edits_end, status_end, notes_end = count_revisions_to_cutoff(session, pageid_int, cutoff_end)

        # Compute delta if both sides are numeric
        delta = ""
        try:
            s_val = int(edits_start) if edits_start == edits_start else None
            e_val = int(edits_end) if edits_end is not None else None
            if s_val is not None and e_val is not None:
                delta = str(e_val - s_val)
        except Exception:
            delta = ""

        _append_row(
            out_csv,
            {
                "season": season,
                "celebrityname": name,
                "wiki_title": wiki_title,
                "pageid": pageid_int,
                "cutoff_start_utc": cutoff_start,
                "edits_start_to_cutoff": edits_start,
                "cutoff_end_utc": cutoff_end,
                "edits_end_to_cutoff": edits_end if edits_end is not None else "",
                "edits_change": delta,
                "status_start": status_start,
                "status_end": status_end,
                "notes": f"{notes_start} {notes_end}".strip(),
            },
        )
        done.add((season, name))


if __name__ == "__main__":
    main()


