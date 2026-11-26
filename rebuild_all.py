import base64
import datetime
import html
import json
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUSHER_HTML = ROOT / "Push_To_GitHub_S24_Ultra.html"
ZIP_OUT = ROOT / "SnipVR.IQ.zip"

EXCLUDE_DIRS = {".venv", "venv", "dist", "build", "python_runtime", "__pycache__", ".git", "ffmpeg_tmp", "platform_tools", "platform_tools_tmp"}
EXCLUDE_FILES = {".DS_Store", "SnipVR.IQ.zip", "VR_Auto_Trimmer_Windows.zip", "passwords.json"}
EXCLUDE_EXTS = {".pyc", ".pyo"}


def collect_files() -> dict[str, str]:
    embedded: dict[str, str] = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name in EXCLUDE_FILES or p.suffix.lower() in EXCLUDE_EXTS:
            continue
        rel = p.relative_to(ROOT).as_posix()
        embedded[rel] = base64.b64encode(p.read_bytes()).decode("ascii")
    return embedded


def main() -> None:
    if len(sys.argv) > 1:
        commit_msg = " ".join(sys.argv[1:]).strip()
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"Release SnipVR.IQ ({stamp})"

    # 1. Embed tracked files & commit message into Push_To_GitHub_S24_Ultra.html
    # (Exclude Push_To_GitHub_S24_Ultra.html itself from EMBEDDED_FILES to avoid self-recursion)
    embedded = {k: v for k, v in collect_files().items() if k != "Push_To_GitHub_S24_Ultra.html"}
    html_text = PUSHER_HTML.read_text(encoding="utf-8")

    embedded_json = json.dumps(embedded, separators=(",", ":"))
    html_text, n1 = re.subn(
        r"const EMBEDDED_FILES = \{.*?\};",
        lambda _m: f"const EMBEDDED_FILES = {embedded_json};",
        html_text,
        count=1,
        flags=re.DOTALL,
    )
    if n1 != 1:
        raise RuntimeError("Could not locate EMBEDDED_FILES placeholder in Push_To_GitHub_S24_Ultra.html")

    escaped_msg = html.escape(commit_msg)
    html_text, n2 = re.subn(
        r'<textarea id="commit-msg">.*?</textarea>',
        lambda _m: f'<textarea id="commit-msg">{escaped_msg}</textarea>',
        html_text,
        count=1,
        flags=re.DOTALL,
    )
    if n2 != 1:
        raise RuntimeError("Could not locate <textarea id=\"commit-msg\"> in Push_To_GitHub_S24_Ultra.html")

    PUSHER_HTML.write_text(html_text, encoding="utf-8")

    # 2. Rebuild companion release archive (includes Push_To_GitHub_S24_Ultra.html)
    all_release_files = collect_files()
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(all_release_files):
            zf.write(ROOT / rel, arcname=f"SnipVR.IQ/{rel}")

    print(f"Embedded {len(embedded)} file(s) into {PUSHER_HTML.name}")
    print(f"Commit message: {commit_msg}")
    print(f"Built {ZIP_OUT.name} ({os.path.getsize(ZIP_OUT)} bytes)")


if __name__ == "__main__":
    main()
