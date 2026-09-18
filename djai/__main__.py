"""Entry point for ``python -m djai``.

THREADING CONTEXT: main thread.
"""

from __future__ import annotations

import sys

from djai.cli import main

if __name__ == "__main__":
    sys.exit(main())
