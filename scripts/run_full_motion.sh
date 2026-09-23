#!/bin/bash
set -euo pipefail
cd /workspace/new4dgs
config=configs/coffee_full_motion.json
output=runs/coffee_full_motion
if /venv/main/bin/python - "$output/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);sys.exit(0 if p.exists() and json.loads(p.read_text()).get('status')=='completed' else 1)
PY
then exit 0; fi
if [[ -f "$output/latest.pt" ]]; then
  exec /venv/main/bin/python -u full_sequence.py --config "$config" --out "$output" --resume "$output/latest.pt"
else
  exec /venv/main/bin/python -u full_sequence.py --config "$config" --out "$output"
fi
