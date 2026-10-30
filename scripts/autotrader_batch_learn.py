"""Run batch historical strategy research. Paper-only; no orders."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.batch_research import main
if __name__=="__main__":main()
