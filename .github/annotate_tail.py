"""Post the useful end of a log file as a GitHub Actions ::error annotation.

Job logs need a GitHub login to download, but annotations are public, so this keeps a
failed step readable through the API. GitHub cuts annotations at about 4 KB, so pip
noise is dropped, error lines go first and the remaining budget is spent on the tail.
Usage: annotate_tail.py TITLE LOGFILE
"""

import re
import sys

BUDGET = 3500
NOISE = re.compile(
    r"Downloading |Collecting |Using cached |Requirement already satisfied|Attempting uninstall|"
    r"Found existing installation|Not uninstalling|Can't uninstall|Stored in directory|━"
)
ERROR = re.compile(r"error|failed|traceback|exception|no matching|not found|cannot|killed", re.IGNORECASE)

title, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8", errors="replace") as f:
    lines = [ln for ln in f.read().splitlines() if ln.strip() and not NOISE.search(ln)]
lines = [ln.encode("ascii", "replace").decode()[:300] for ln in lines]

errors = [ln for ln in lines if ERROR.search(ln)][-8:]
head = "ERROR LINES:\n" + "\n".join(errors) + "\n\nTAIL:\n" if errors else "TAIL:\n"
tail: list[str] = []
used = len(head)
for line in reversed(lines):
    if used + len(line) + 1 > BUDGET:
        break
    tail.insert(0, line)
    used += len(line) + 1
message = (head + "\n".join(tail)).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print(f"::error title={title}::{message}")
