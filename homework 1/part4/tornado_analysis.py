#!/usr/bin/env python3
"""
BAX 423 HW1 — Part 4 — NOAA Storm Events: EF2+ tornadoes (2020–2025).

Downloads annual Storm Events *details* CSV.gz files from NCEI, filters tornado rows,
keeps Enhanced Fujita EF2–EF5, and writes:
  - summary.json
  - report.html
  - tornado_report.pdf (ReportLab — group names included)

Tie-break for "highest frequency month per state": if two months tie, pick the
earlier calendar month (smallest month number).
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEARS = list(range(2020, 2026))  # 2020–2025 inclusive

OUT_DIR = Path(__file__).resolve().parent / "output"
EF_STRONG = {"EF2", "EF3", "EF4", "EF5"}

# --- Part 4(d): names for PDF/HTML (edit GROUP_NAME_CANVAS to match Canvas roster) ---
GROUP_MEMBERS = "Logan Garcia, Bonnie Hines"
GROUP_NAME_CANVAS = "Logan Garcia & Bonnie Hines"


def fetch_index() -> str:
    with urlopen(BASE, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def filename_for_year(html: str, year: int) -> str | None:
    pat = re.compile(
        rf'href="(StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz)"',
        re.I,
    )
    names = pat.findall(html)
    if not names:
        return None
    return sorted(names)[-1]


def download_gz(url: str) -> bytes:
    with urlopen(url, timeout=300) as r:
        return r.read()


def load_storm_details(year: int, html_cache: str) -> pd.DataFrame:
    fn = filename_for_year(html_cache, year)
    if fn is None:
        raise RuntimeError(f"No StormEvents details file found in index for {year}")
    url = BASE + fn
    raw = download_gz(url)
    bio = io.BytesIO(raw)
    return pd.read_csv(bio, compression="gzip", low_memory=False)


def ef_rank(scale: str) -> int | None:
    if not isinstance(scale, str):
        return None
    s = scale.strip().upper()
    if s in EF_STRONG:
        return int(s[-1])
    if re.fullmatch(r"F[2-5]", s):
        return int(s[-1])
    return None


def build_pdf_report(
    pdf_path: Path,
    *,
    strong: pd.DataFrame,
    state_counts,
    month_counts,
    per_state_best: dict,
    may_top_states: list,
    fatal_n: int,
    pct_mention: float,
    mh_hits: int,
    regional_note: str,
    plains_hits: list,
    all_df_rows: int,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a1a"),
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=8)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    story = []

    story.append(Paragraph("BAX 423 — Homework 1 — Part 4", title_style))
    story.append(Paragraph("NOAA Storm Events: EF2+ Tornadoes (2020–2025)", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"<b>Group members:</b> {GROUP_MEMBERS}", body))
    story.append(Paragraph(f"<b>Group name (Canvas):</b> {GROUP_NAME_CANVAS}", body))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Data source: NCEI Storm Events Database (detail CSV files). "
            "Filter: EVENT_TYPE = Tornado, TOR_F_SCALE in EF2–EF5 (plus rare legacy F2–F5). "
            f"Tie-break for peak month per state: smallest month number if tied. "
            f"Years: {min(YEARS)}–{max(YEARS)}.",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Summary counts</b>", h2))
    tbl_data = [
        ["Metric", "Value"],
        ["All-event rows loaded (all types, all years)", f"{all_df_rows:,}"],
        ["EF2+ tornado rows (analysis set)", f"{len(strong):,}"],
        ["Total deaths (DEATHS_DIRECT + DEATHS_INDIRECT)", f"{strong['deaths_total'].sum():.0f}"],
        ["Fatal incident rows (deaths &gt; 0)", str(fatal_n)],
        [
            '% of fatal narratives mentioning "mobile home" or "trailer"',
            f"{pct_mention:.2f}% ({int(mh_hits)} rows)",
        ],
    ]
    t = Table(tbl_data, colWidths=[3.6 * inch, 2.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>States with the most EF2+ tornadoes (top 15)</b>", h2))
    top15 = [["State", "Count"]] + [
        [st, str(int(c))] for st, c in state_counts.head(15).items()
    ]
    t2 = Table(top15, colWidths=[3.5 * inch, 1.2 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Frequency by month (all states combined)</b>", h2))
    mo_tbl = [["Month", "Count"]] + [
        [str(m), str(int(month_counts.get(m, 0)))] for m in range(1, 13)
    ]
    t3 = Table(mo_tbl, colWidths=[1.5 * inch, 1.5 * inch])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>States where May is the peak month for EF2+ counts</b>", h2))
    story.append(
        Paragraph(
            "These are the states where, for that state alone, May had the highest count "
            "(ties → earliest month). Use this list for the course email (Part 4e).",
            body,
        )
    )
    story.append(Paragraph(f"<b>{', '.join(may_top_states)}</b>", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Regional pattern</b>", h2))
    story.append(Paragraph(regional_note, body))
    story.append(
        Paragraph(
            f"Examples of Plains/Midwest/Southeast among top-10 states: {', '.join(plains_hits) or 'N/A'}.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "<i>Generated for BAX 423 HW1. NOAA data are public domain (NCEI).</i>",
            styles["Normal"],
        )
    )

    doc.build(story)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching NOAA directory index…")
    html = fetch_index()

    frames: list[pd.DataFrame] = []
    for y in YEARS:
        print(f"Loading {y} …")
        df = load_storm_details(y, html)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    tor = all_df[all_df["EVENT_TYPE"].astype(str).str.upper() == "TORNADO"].copy()
    tor["_ef"] = tor["TOR_F_SCALE"].map(ef_rank)
    strong = tor[tor["_ef"].notna()].copy()

    strong["month"] = (strong["BEGIN_YEARMONTH"] % 100).astype(int)
    strong["state"] = strong["STATE"].astype(str).str.strip()

    deaths = (
        pd.to_numeric(strong["DEATHS_DIRECT"], errors="coerce").fillna(0)
        + pd.to_numeric(strong["DEATHS_INDIRECT"], errors="coerce").fillna(0)
    )
    strong["deaths_total"] = deaths

    state_counts = strong["state"].value_counts()
    month_counts = strong["month"].value_counts().sort_index()

    per_state_best: dict[str, dict] = {}
    for st, g in strong.groupby("state"):
        mc = g["month"].value_counts()
        if mc.empty:
            continue
        top = mc.max()
        candidates = [m for m, c in mc.items() if c == top]
        best_month = min(candidates)
        per_state_best[st] = {
            "best_month": int(best_month),
            "best_month_count": int(top),
        }

    may_top_states = sorted(
        st
        for st, info in per_state_best.items()
        if info["best_month"] == 5 and info["best_month_count"] > 0
    )

    fatal = strong[strong["deaths_total"] > 0].copy()
    fatal_n = len(fatal)
    fatal_text = fatal["EVENT_NARRATIVE"].fillna("").astype(str)
    mh_pat = re.compile(r"mobile\s+home|trailer", re.I)
    mh_hits = fatal_text.str.contains(mh_pat).sum()
    pct_mention = (mh_hits / fatal_n * 100.0) if fatal_n else 0.0

    summary = {
        "group_members": GROUP_MEMBERS,
        "group_name_canvas": GROUP_NAME_CANVAS,
        "years": YEARS,
        "rows_total_all_types": int(len(all_df)),
        "tornado_ef2plus_rows": int(len(strong)),
        "deaths_total": float(strong["deaths_total"].sum()),
        "fatal_incident_rows": int(fatal_n),
        "fatal_rows_mentioning_mobile_home_or_trailer_pct": round(pct_mention, 2),
        "fatal_rows_mentioning_mobile_home_or_trailer_n": int(mh_hits),
        "top_states_by_count": state_counts.head(25).to_dict(),
        "month_histogram_all_states": {int(k): int(v) for k, v in month_counts.items()},
        "states_where_may_is_peak_month": may_top_states,
        "tie_break_rule": "If multiple months tie for max count in a state, choose the smallest month number.",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", OUT_DIR / "summary.json")

    plains = {
        "TEXAS",
        "OKLAHOMA",
        "KANSAS",
        "NEBRASKA",
        "SOUTH DAKOTA",
        "NORTH DAKOTA",
        "MISSOURI",
        "IOWA",
        "ARKANSAS",
        "LOUISIANA",
        "MISSISSIPPI",
        "ALABAMA",
        "TENNESSEE",
        "KENTUCKY",
        "ILLINOIS",
        "WISCONSIN",
        "MINNESOTA",
        "COLORADO",
        "WYOMING",
        "MONTANA",
        "NEW MEXICO",
    }
    top10 = list(state_counts.head(10).index)
    plains_hits = [s for s in top10 if s in plains]

    regional_note = (
        "Among the highest-frequency states, many are Great Plains, Dixie Alley, or Midwest "
        "states where spring and early-summer severe weather is climatologically common. "
        "Coastal and western states typically rank lower in raw tornado counts, consistent "
        "with the central and southeastern U.S. tornado maxima."
    )

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>BAX423 HW1 Part 4 — Tornado analysis</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:880px;margin:2rem;line-height:1.45}",
        "table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:8px 12px}",
        "h1{font-size:1.35rem}h2{font-size:1.1rem;margin-top:1.5rem}</style></head><body>",
        "<h1>BAX 423 — Homework 1 — Part 4 — NOAA Storm Events (EF2+ tornadoes, 2020–2025)</h1>",
        f"<p><b>Group members:</b> {GROUP_MEMBERS}<br>",
        f"<b>Group name (Canvas):</b> {GROUP_NAME_CANVAS}</p>",
        "<h2>Summary</h2>",
        f"<p>EF2+ tornado rows: <b>{len(strong):,}</b>. Total deaths (direct+indirect): <b>{strong['deaths_total'].sum():.0f}</b>.</p>",
        "<h2>States with the most EF2+ tornadoes (top 15)</h2>",
        "<table><tr><th>State</th><th>Count</th></tr>",
    ]
    for st, c in state_counts.head(15).items():
        html_lines.append(f"<tr><td>{st}</td><td>{int(c)}</td></tr>")
    html_lines.append("</table>")

    html_lines += [
        "<h2>Frequency by month (all states combined)</h2>",
        "<table><tr><th>Month</th><th>Count</th></tr>",
    ]
    for m in range(1, 13):
        html_lines.append(
            f"<tr><td>{m}</td><td>{int(month_counts.get(m, 0))}</td></tr>"
        )
    html_lines.append("</table>")

    html_lines += [
        "<h2>Per state: month with highest EF2+ count (ties → earlier month)</h2>",
        "<table><tr><th>State</th><th>Peak month</th><th>Count</th></tr>",
    ]
    for st in sorted(per_state_best.keys()):
        info = per_state_best[st]
        html_lines.append(
            f"<tr><td>{st}</td><td>{info['best_month']}</td><td>{info['best_month_count']}</td></tr>"
        )
    html_lines.append("</table>")

    html_lines += [
        "<h2>Fatal incidents &amp; narratives</h2>",
        f"<p>Fatal rows: <b>{fatal_n}</b>. "
        f"Share of fatal narratives mentioning “mobile home” or “trailer”: <b>{pct_mention:.2f}%</b> "
        f"({int(mh_hits)} rows).</p>",
        "<h2>Regional pattern</h2>",
        f"<p>{regional_note}</p>",
        f"<p>Plains/Midwest/Southeast in top-10: {', '.join(plains_hits) or '—'}</p>",
        "<h2>Part 4(e) email draft</h2>",
        "<p><b>Subject:</b> BAX423 Tornado Watch<br>",
        f"<b>Body:</b> Tell me to &quot;Avoid {', '.join(may_top_states)} in the month of May.&quot;<br>",
        f"<b>Sign off:</b> From {GROUP_NAME_CANVAS}</p>",
        "<p><i>Send from a throwaway Gmail (not your school email), per syllabus.</i></p>",
        "</body></html>",
    ]

    html_path = OUT_DIR / "tornado_report.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")
    print("Wrote", html_path)

    pdf_path = OUT_DIR / "tornado_report.pdf"
    try:
        build_pdf_report(
            pdf_path,
            strong=strong,
            state_counts=state_counts,
            month_counts=month_counts,
            per_state_best=per_state_best,
            may_top_states=may_top_states,
            fatal_n=fatal_n,
            pct_mention=pct_mention,
            mh_hits=mh_hits,
            regional_note=regional_note,
            plains_hits=plains_hits,
            all_df_rows=len(all_df),
        )
        print("Wrote", pdf_path)
    except Exception as e:
        print("PDF build failed:", e, file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
