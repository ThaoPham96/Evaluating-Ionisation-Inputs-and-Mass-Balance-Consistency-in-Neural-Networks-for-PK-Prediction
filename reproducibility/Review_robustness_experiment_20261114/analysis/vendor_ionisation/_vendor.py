"""Locate the immutable, attributed mass-balance dependency snapshot."""
from pathlib import Path
import sys


def ensure_vendor():
    path = str(Path(__file__).resolve().parent / 'vendor_mass_balance')
    if path not in sys.path:
        sys.path.append(path)
    return Path(path)
