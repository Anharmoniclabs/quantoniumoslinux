"""
QuantDiff CLI — stdin JSON → stdout JSON.

Same wire protocol as src.resonart.cli so the Rust backend can spawn it
identically. (No external diffusers backend — fully native Resonart.)

    python -m src.resonart.quantdiff.cli  < request.json
"""

from __future__ import annotations

import json
import sys

from src.aethercanvas.engine import GenerateRequest
from .engine import QuantDiffEngine

_FIELDS = set(GenerateRequest.__dataclass_fields__.keys())


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"success": False, "error": "empty stdin"}))
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "error": f"JSON parse error: {exc}"}))
        sys.exit(1)

    req = GenerateRequest(**{k: v for k, v in data.items() if k in _FIELDS})
    out_dir = data.get("output_dir", "generated/images")

    engine = QuantDiffEngine(output_dir=out_dir)
    result = engine.generate(req)

    print(
        json.dumps(
            {
                "success": result.success,
                "path": result.path,
                "prompt": result.prompt,
                "style": result.style,
                "seed": result.seed,
                "width": result.width,
                "height": result.height,
                "error": result.error,
            }
        )
    )


if __name__ == "__main__":
    main()
