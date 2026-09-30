"""Song -> vocals -> pitch (f0) -> MIDI notes -> piano render + interactive piano roll."""

import os

# Let ops that MPS doesn't implement fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
