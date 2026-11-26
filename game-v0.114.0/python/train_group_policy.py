"""Run a grouped policy experiment against the official headless game host."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sts2rl.group_train import main

if __name__ == "__main__":
    main()
