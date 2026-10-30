"""CLI for local historical daily-candle evidence; does not trade."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import main
if __name__=="__main__":main()
