"""Run the CIFAR-100 evaluate command."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cifar100.cli import evaluate_main

if __name__ == "__main__":
    evaluate_main()
