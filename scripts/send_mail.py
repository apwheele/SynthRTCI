"""Email the result of a data refresh (.github/workflows/pages.yml).

Sends through Titan mail as andrew.wheeler@crimede-coder.com. The password
comes from the SMTP_PASSWORD environment variable, a repository secret set by
CrimeDecomp's src/release_mail.ps1 -Setup (or `gh secret set SMTP_PASSWORD`).

    python scripts/send_mail.py --subject "..." --body "..." [--attach run.log]
"""

import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

ADDRESS = "andrew.wheeler@crimede-coder.com"
SERVER, PORT = "smtp.titan.email", 465


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--attach", action="append", default=[], help="text file to attach, if it exists")
    args = ap.parse_args()

    msg = EmailMessage()
    msg["From"] = ADDRESS
    msg["To"] = ADDRESS
    msg["Subject"] = args.subject
    msg.set_content(args.body)
    for path in map(Path, args.attach):
        if path.is_file():
            msg.add_attachment(path.read_bytes(), maintype="text", subtype="plain", filename=path.name)

    with smtplib.SMTP_SSL(SERVER, PORT, context=ssl.create_default_context(), timeout=60) as smtp:
        smtp.login(ADDRESS, os.environ["SMTP_PASSWORD"].strip())
        smtp.send_message(msg)
    print(f"Sent: {args.subject}")


if __name__ == "__main__":
    main()
