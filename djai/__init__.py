"""djai - an AI-mixed DJ.

Three strictly separated layers:

1. Chat / intent layer (:mod:`djai.intent`) -- calls an LLM, produces structured
   commands. Runs on the REPL thread. NEVER in the audio path.
2. Command queue (:mod:`djai.commands`, :mod:`djai.scheduler`) -- structured
   commands quantized to musical position. Runs on the scheduler thread.
3. Deterministic real-time audio engine (:mod:`djai.deck`, :mod:`djai.engine`,
   :mod:`djai.transition`) -- the audio callback. No LLM, no I/O, no blocking.

Threading context is documented at the top of every module.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
