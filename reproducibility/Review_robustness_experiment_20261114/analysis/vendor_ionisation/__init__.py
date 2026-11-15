"""Expose the exact local dependencies used by the archived ionisation adapters."""

from pathlib import Path
import sys


_root = Path(__file__).resolve().parent
for _directory in (_root/'vendor_mass_balance', _root):
    _path = str(_directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)
