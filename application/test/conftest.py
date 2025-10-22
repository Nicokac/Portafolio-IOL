import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Ensure project root is available for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
