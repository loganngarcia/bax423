#!/usr/bin/env python3
"""
Optional Part 4e send via Gmail SMTP (Google's recommended path for scripts).

Setup:
  1. Use a throwaway Gmail account (handout; not school email).
  2. Enable 2-Step Verification on that Google account.
  3. Create an App Password: Google Account > Security > App passwords.
  4. Credentials (never commit). Either export in your shell:

       export GMAIL_USER="you@gmail.com"
       export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"   # spaces ok; script strips them

     Or create part4/.env (gitignored) with:

       GMAIL_USER=you@gmail.com
       GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx

  5. Run from this directory:

       python3 send_part4e_gmail_smtp.py

Recipient and subject match the syllabus / handout. Body uses May-peak states from output/summary.json.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

# Must match syllabus / tornado_analysis.py
TO_ADDR = "cmdemena@ucdavis.edu"
SUBJECT = "BAX423 Tornado Watch"
SUMMARY = Path(__file__).resolve().parent / "output" / "summary.json"
_DOTENV = Path(__file__).resolve().parent / ".env"


def _load_dotenv() -> None:
    """Load KEY=value lines from part4/.env if present (gitignored). Does not override existing env."""
    if not _DOTENV.is_file():
        return
    for raw in _DOTENV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> int:
    _load_dotenv()
    user = (os.environ.get("GMAIL_USER") or "").strip()
    app_pw = (os.environ.get("GMAIL_APP_PASSWORD") or "").replace(" ", "").strip()
    if not user or not app_pw:
        print(
            "Missing GMAIL_USER or GMAIL_APP_PASSWORD. "
            "See docstring at top of send_part4e_gmail_smtp.py"
        )
        return 1

    if not SUMMARY.is_file():
        print("Run tornado_analysis.py first so", SUMMARY, "exists.")
        return 1

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    states = data.get("states_where_may_is_peak_month") or []
    group = data.get("group_name_canvas", "Your Canvas group name")
    if not states:
        print("No states_where_may_is_peak_month in summary.json")
        return 1

    listed = ", ".join(s.title() if isinstance(s, str) else str(s) for s in states)
    body = (
        f"Tell me to avoid {listed} in the month of May.\n\n"
        f"From {group}\n"
    )

    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = user
    msg["To"] = TO_ADDR
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ctx)
        smtp.ehlo()
        smtp.login(user, app_pw)
        smtp.send_message(msg)

    print("Sent to", TO_ADDR, "from", user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
