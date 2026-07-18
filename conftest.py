"""Make pytest import the local `eos` package, not the site-packages install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
