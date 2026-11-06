"""Render an inference checkpoint or an independently decoded .adagc stream."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from adasplat.codec.cli import main
if __name__=='__main__':raise SystemExit(main(['render']+sys.argv[1:]))
