"""Play the best trained FireShield agent with the GUI.

Same as `uv run main.py play`; accepts the same options, e.g.
    uv run play.py --algo ppo --episodes 5
"""

import sys

from main import main

if __name__ == "__main__":
    main(["play", *sys.argv[1:]])
