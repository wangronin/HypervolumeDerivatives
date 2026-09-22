"""Tune MMD Newton with point matching; see tune_MMD.py for common options."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tune_MMD import main


if __name__ == "__main__":
    main(matching=True)
