"""Run the CIFAR-100 predict command."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cifar100.cli import predict_main

if __name__ == "__main__":
    predict_main()
