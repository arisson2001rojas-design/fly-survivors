"""Download the FlyWire v783 connectome files used by the model.

The files are those redistributed with Shiu et al. 2024
(https://github.com/philshiu/Drosophila_brain_model, MIT). The underlying FlyWire data is
subject to the FlyWire data license (non-commercial, attribution); see README.
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
FILES = ["Completeness_783.csv", "Connectivity_783.parquet"]
DEST = Path(__file__).resolve().parents[1] / "data" / "flywire_783"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists():
            print(f"ok      {target} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"fetch   {BASE + name}")
        urllib.request.urlretrieve(BASE + name, target)
        print(f"saved   {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
