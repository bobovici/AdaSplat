"""Single-image entry point for the shared Trainer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from adasplat.training.cli import single_main

if __name__=='__main__':
    raise SystemExit(single_main())
