"""Shared snapshot writer for the Plan 13 page. Callers commit the volume."""
import json
import os
import time
from pathlib import Path
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

LIVE = Path(f"{_DATA}/plan13/live")
EVENTS = LIVE / "events.json"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def snapshot(name, payload):
    payload = dict(payload)
    payload["updated_ts"] = time.time()
    write_json(LIVE / f"{name}.json", payload)


def add_event(text):
    rows = []
    if EVENTS.is_file():
        try:
            rows = json.loads(EVENTS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = []
    if rows and rows[0].get("text") == text:
        return
    rows.insert(0, {"t": time.time(), "text": text})
    write_json(EVENTS, rows[:50])
