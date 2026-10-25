#!/usr/bin/env python3
"""CLI entrypoint for the Resonart engine."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.aethercanvas.engine import ResonartEngine, GenerateRequest


def main() -> None:
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"success": False, "error": "No input received"}))
            return
        data = json.loads(raw)
        req = GenerateRequest(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            steps=int(data.get("steps", 20)),
            style=data.get("style", "photo"),
            width=int(data.get("width", 512)),
            height=int(data.get("height", 512)),
            seed=data.get("seed"),
            output_dir=data.get("output_dir", "generated/images"),
        )
        if not req.prompt.strip():
            print(json.dumps({"success": False, "error": "Prompt is empty"}))
            return
        result = AetherCanvasEngine(output_dir=req.output_dir).generate(req)
        print(json.dumps({
            "success": result.success,
            "path": result.path,
            "prompt": result.prompt,
            "style": result.style,
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "error": result.error,
        }))
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "error": f"JSON parse error: {exc}"}))
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))


if __name__ == "__main__":
    main()