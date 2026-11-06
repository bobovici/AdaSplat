"""Batch scheduling entry point; each child runs the same single-image Trainer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from adasplat.training.cli import batch_main

if __name__=='__main__':
    raise SystemExit(batch_main())
