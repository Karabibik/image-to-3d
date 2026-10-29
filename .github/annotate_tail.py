"""Print the tail of a log file as a GitHub Actions ::error annotation.

Job logs need a GitHub login to download, but annotations are public, so this keeps a
failed step readable through the API. Usage: annotate_tail.py TITLE LOGFILE [LINES]
"""

import sys

title, path = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 60
with open(path, encoding="utf-8", errors="replace") as f:
    tail = f.read().splitlines()[-count:]
message = "\n".join(tail).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print(f"::error title={title}::{message}")
