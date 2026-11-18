"""Part D decisions and live counters. No GPU and no network."""
import time
from collections import Counter

from plan12_letter import PRIORITY, letter_score, letters_of
from plan12_text import normalize_compare

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
L40S_HOUR = 1.95
REGION_FACTOR = 1.15
FULL_DIALECTS = PRIORITY | {"ar_irq", "ar_lev", "ar_omn"}
BORDER_ONLY = {"ar_dza", "ar_mar", "ar_tun"}
FULL_TIERS = {"Borderline", "Tier 2", "Tier 3"}
COST_PER_GPU_HOUR = L40S_HOUR * REGION_FACTOR


def dialect_label(value) -> str:
    """Empty and plain ar are unknown. Every other value stays as written."""
    text = str(value or "").strip()
    if not text or text == "ar":
        return "unknown"
    return text


def choose_vllm(hf_rate: float, vllm_rate: float) -> bool:
    """vLLM wins when its mean letter rate is at most 0.5 points above HF."""
    return float(vllm_rate) <= float(hf_rate) + 0.005


def keep_speedup(identical: float, baseline_rate: float, candidate_rate: float) -> bool:
    """99% of outputs match after normalisation, and Jais letter rate is within 0.2 points."""
    return float(identical) >= 0.99 and float(candidate_rate) <= float(baseline_rate) + 0.002


def eligible(dialect: str, tier: str) -> bool:
    if dialect in BORDER_ONLY:
        return tier == "Borderline"
    if dialect in FULL_DIALECTS:
        return tier in FULL_TIERS
    return False


def promoted(jais: str, cohere: str) -> bool:
    return letter_score(jais, cohere)["letter_rate"] <= 0.03


def pair_rates(jais: str, audar: str, cohere: str) -> dict:
    ja = letter_score(jais, audar)["letter_rate"]
    jc = letter_score(jais, cohere)["letter_rate"]
    ac = letter_score(audar, cohere)["letter_rate"]
    distances = {"jais": ja + jc, "audar": ja + ac, "cohere": jc + ac}
    best = max(distances.values())
    names = [name for name, value in distances.items() if value == best]
    odd = names[0] if len(names) == 1 and best > 0 else "tie"
    return {"ja": ja, "jc": jc, "ac": ac, "odd": odd}


def identical_norm(left: str, right: str) -> bool:
    return letters_of(left) == letters_of(right)


def substitutions(ref: str, hyp: str):
    """Word substitutions after normalize_compare. Insertions and deletions are not counted."""
    a = normalize_compare(ref).split()
    b = normalize_compare(hyp).split()
    n, m = len(a), len(b)
    if n == 0 or m == 0 or n * m > 2500:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        prev = dp[i - 1]
        row = dp[i]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j], row[j - 1], prev[j - 1])
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            out.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def dollars(gpu_seconds: float) -> float:
    return round(float(gpu_seconds) / 3600 * COST_PER_GPU_HOUR, 2)


def project_cost(spent_seconds: float, done: int, total: int) -> float:
    if done <= 0 or total <= done:
        return dollars(spent_seconds)
    return dollars(spent_seconds * total / done)


class Tally:
    """Running Part D stats. Snapshot is small enough to write every two minutes."""

    def __init__(self):
        self.clips = 0
        self.hours = 0.0
        self.sha_skips = 0
        self.empty = 0
        self.cohere_done = 0
        self.dialects = {}
        self.sources = {}
        self.swaps = Counter()
        self.examples = {}
        self.audio_s = 0.0
        self.gpu_s = 0.0
        self.samples = []

    def _dialect(self, name):
        return self.dialects.setdefault(name, {
            "clips": 0, "hours": 0.0, "promoted_hours": 0.0, "promoted_clips": 0,
            "tier3_hours": 0.0, "ja": 0.0, "jc": 0.0, "ac": 0.0, "n": 0,
            "odd": Counter(), "buckets": {}, "short_n": 0, "short_sum": 0.0,
            "long_n": 0, "long_sum": 0.0,
        })

    def add_scored(self, row):
        """row needs dialect, duration, jais, audar, cohere, tier, source, words."""
        self.clips += 1
        self.cohere_done += 1
        duration = float(row.get("duration") or 0)
        self.hours += duration / 3600
        if not str(row.get("cohere") or "").strip():
            self.empty += 1
        dialect = row.get("dialect") or "(none)"
        slot = self._dialect(dialect)
        slot["clips"] += 1
        slot["hours"] += duration / 3600
        rates = pair_rates(row.get("jais") or "", row.get("audar") or "", row.get("cohere") or "")
        slot["ja"] += rates["ja"]
        slot["jc"] += rates["jc"]
        slot["ac"] += rates["ac"]
        slot["n"] += 1
        slot["odd"][rates["odd"]] += 1
        is_promoted = bool(row.get("promoted"))
        if is_promoted:
            slot["promoted_hours"] += duration / 3600
            slot["promoted_clips"] += 1
        tier = row.get("tier") or ""
        if tier == "Tier 3" and not is_promoted:
            slot["tier3_hours"] += duration / 3600
        bucket = f"{int(duration // 2) * 2}-{int(duration // 2) * 2 + 2}"
        band = slot["buckets"].setdefault(bucket, {"n": 0, "sum": 0.0})
        band["n"] += 1
        band["sum"] += rates["jc"]
        words = int(row.get("words") or 0)
        if words < 5:
            slot["short_n"] += 1
            slot["short_sum"] += rates["jc"]
        else:
            slot["long_n"] += 1
            slot["long_sum"] += rates["jc"]
        source = row.get("source") or "(none)"
        src = self.sources.setdefault(source, {"hours": 0.0, "tier3_hours": 0.0})
        src["hours"] += duration / 3600
        if tier == "Tier 3" and not is_promoted:
            src["tier3_hours"] += duration / 3600
        if dialect in PRIORITY:
            for left, right in substitutions(row.get("jais") or "", row.get("cohere") or "")[:8]:
                self.swaps[(dialect, left, right)] += 1
            bag = self.examples.setdefault(dialect, {"promoted": [], "kept": []})
            key = "promoted" if is_promoted else "kept"
            bag[key].append({
                "clip_id": row.get("clip_id") or "",
                "jais": row.get("jais") or "",
                "audar": row.get("audar") or "",
                "cohere": row.get("cohere") or "",
                "t": row.get("t") or "",
            })
            bag[key] = bag[key][-3:]

    def note_skip(self):
        self.sha_skips += 1

    def note_time(self, audio_s=0.0, gpu_s=0.0):
        self.audio_s += float(audio_s)
        self.gpu_s += float(gpu_s)

    def mark(self, done, now=None):
        now = time.time() if now is None else now
        self.samples.append({"t": now, "done": int(done)})
        self.samples = [item for item in self.samples if now - item["t"] <= 900]

    def snapshot(self, name, state, done, total, gpu_mem_mb=0.0, extra=None):
        swaps = {}
        for dialect in PRIORITY:
            pairs = [(left, right, count) for (dia, left, right), count in self.swaps.items() if dia == dialect]
            pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
            swaps[dialect] = pairs[:50]
        dialects = {}
        for dialect, slot in self.dialects.items():
            dialects[dialect] = {
                "clips": slot["clips"],
                "hours": round(slot["hours"], 4),
                "promoted_hours": round(slot["promoted_hours"], 4),
                "promoted_clips": slot["promoted_clips"],
                "tier3_hours": round(slot["tier3_hours"], 4),
                "ja": slot["ja"], "jc": slot["jc"], "ac": slot["ac"], "n": slot["n"],
                "odd": dict(slot["odd"]),
                "buckets": {key: {"n": value["n"], "sum": value["sum"]} for key, value in slot["buckets"].items()},
                "short_n": slot["short_n"], "short_sum": slot["short_sum"],
                "long_n": slot["long_n"], "long_sum": slot["long_sum"],
            }
        payload = {
            "name": name,
            "state": state,
            "done": int(done),
            "total": int(total),
            "hours_done": round(self.hours, 4),
            "sha_skips": self.sha_skips,
            "empty": self.empty,
            "cohere_done": self.cohere_done,
            "audio_s": round(self.audio_s, 3),
            "gpu_s": round(self.gpu_s, 3),
            "gpu_mem_mb": round(float(gpu_mem_mb), 1),
            "samples": self.samples[-12:],
            "dialects": dialects,
            "sources": {key: {"hours": round(value["hours"], 4), "tier3_hours": round(value["tier3_hours"], 4)}
                        for key, value in self.sources.items()},
            "swaps": swaps,
            "examples": self.examples,
            "updated_ts": time.time(),
        }
        if extra:
            payload.update(extra)
        return payload


def _rate_between(older, newer):
    span = newer["t"] - older["t"]
    gained = newer["done"] - older["done"]
    if span <= 0 or gained < 0:
        return None
    return gained / span * 60


def worker_alerts(snapshot, now=None):
    """Red lines for one worker. Promotion and spend are checked on the combined page."""
    now = time.time() if now is None else now
    found = []
    name = snapshot.get("name") or "worker"
    if snapshot.get("state") == "running" and now - float(snapshot.get("updated_ts") or now) > 180:
        found.append(f"{name} has not updated for 3 minutes")
    samples = snapshot.get("samples") or []
    recent = [item for item in samples if now - item["t"] <= 300]
    prior = [item for item in samples if 300 < now - item["t"] <= 600]
    if len(recent) >= 2 and len(prior) >= 2:
        now_rate = _rate_between(recent[0], recent[-1])
        old_rate = _rate_between(prior[0], prior[-1])
        if now_rate is not None and old_rate and now_rate < 0.5 * old_rate:
            found.append(f"{name} slowed by more than half for 5 minutes")
    seen = int(snapshot.get("done") or 0) + int(snapshot.get("sha_skips") or 0)
    if seen and int(snapshot.get("sha_skips") or 0) / seen > 0.005:
        found.append(f"{name} SHA-1 skips are above 0.5%")
    cohere_done = int(snapshot.get("cohere_done") or 0)
    if cohere_done and int(snapshot.get("empty") or 0) / cohere_done > 0.01:
        found.append(f"{name} empty Cohere outputs are above 1%")
    return found


def promotion_alerts(dialects):
    found = []
    for dialect, slot in dialects.items():
        if dialect not in PRIORITY:
            continue
        hours = float(slot.get("hours") or 0)
        if hours < 1:
            continue
        rate = float(slot.get("promoted_hours") or 0) / hours
        if rate < 0.10 or rate > 0.90:
            found.append(f"{dialect} promotion is {round(rate * 100, 1)}% after {round(hours, 2)} h")
    return found


def spend_alert(spent: float, approved: float):
    if not approved:
        return []
    if float(spent) > float(approved) * 1.2:
        return [f"Spend ${spent:.2f} passed the approved ${approved:.2f} by more than 20%"]
    return []


def combine_workers(snapshots):
    """Sum worker snapshots for the page. The page does not rescore clips."""
    dialects = {}
    sources = {}
    swaps = Counter()
    examples = {}
    done = hours = sha = empty = cohere = audio = gpu = 0
    total = 0
    for snap in snapshots:
        done += int(snap.get("done") or 0)
        total += int(snap.get("total") or 0)
        hours += float(snap.get("hours_done") or 0)
        sha += int(snap.get("sha_skips") or 0)
        empty += int(snap.get("empty") or 0)
        cohere += int(snap.get("cohere_done") or 0)
        audio += float(snap.get("audio_s") or 0)
        gpu += float(snap.get("gpu_s") or 0)
        for dialect, slot in (snap.get("dialects") or {}).items():
            dest = dialects.setdefault(dialect, {
                "clips": 0, "hours": 0.0, "promoted_hours": 0.0, "promoted_clips": 0,
                "tier3_hours": 0.0, "ja": 0.0, "jc": 0.0, "ac": 0.0, "n": 0,
                "odd": Counter(), "buckets": {}, "short_n": 0, "short_sum": 0.0,
                "long_n": 0, "long_sum": 0.0,
            })
            for key in ("clips", "promoted_clips", "n", "short_n", "long_n"):
                dest[key] += int(slot.get(key) or 0)
            for key in ("hours", "promoted_hours", "tier3_hours", "ja", "jc", "ac", "short_sum", "long_sum"):
                dest[key] += float(slot.get(key) or 0)
            dest["odd"].update(slot.get("odd") or {})
            for bucket, band in (slot.get("buckets") or {}).items():
                got = dest["buckets"].setdefault(bucket, {"n": 0, "sum": 0.0})
                got["n"] += int(band.get("n") or 0)
                got["sum"] += float(band.get("sum") or 0)
        for source, slot in (snap.get("sources") or {}).items():
            dest = sources.setdefault(source, {"hours": 0.0, "tier3_hours": 0.0})
            dest["hours"] += float(slot.get("hours") or 0)
            dest["tier3_hours"] += float(slot.get("tier3_hours") or 0)
        for dialect, pairs in (snap.get("swaps") or {}).items():
            for left, right, count in pairs:
                swaps[(dialect, left, right)] += int(count)
        for dialect, bag in (snap.get("examples") or {}).items():
            dest = examples.setdefault(dialect, {"promoted": [], "kept": []})
            for key in ("promoted", "kept"):
                dest[key].extend(bag.get(key) or [])
                dest[key] = dest[key][-3:]
    return {
        "done": done, "total": total, "hours_done": hours, "sha_skips": sha,
        "empty": empty, "cohere_done": cohere, "audio_s": audio, "gpu_s": gpu,
        "dialects": dialects, "sources": sources, "swaps": swaps, "examples": examples,
    }


def _selftest():
    assert dialect_label("") == "unknown"
    assert dialect_label("ar") == "unknown"
    assert dialect_label("ar-egy") == "ar-egy"
    assert choose_vllm(0.10, 0.105)
    assert not choose_vllm(0.10, 0.106)
    assert keep_speedup(0.99, 0.10, 0.102)
    assert not keep_speedup(0.989, 0.10, 0.102)
    assert not keep_speedup(0.99, 0.10, 0.103)
    assert eligible("ar_egy", "Tier 2")
    assert not eligible("ar_egy", "Tier 1")
    assert eligible("ar_dza", "Borderline")
    assert not eligible("ar_dza", "Tier 2")
    assert promoted("سلام", "سلام")
    assert not promoted("سلام عليكم", "شيء مختلف تماما هنا الآن")
    assert identical_norm("سلام.", "سلام")
    subs = substitutions("الكتاب على الطاولة", "الكتاب فوق الطاولة")
    assert ("علي", "فوق") in subs
    rates = pair_rates("سلام", "سلام", "سلام")
    assert rates["odd"] == "tie"
    tally = Tally()
    tally.add_scored({
        "dialect": "ar_egy", "duration": 4, "jais": "سلام", "audar": "سلام", "cohere": "سلام",
        "tier": "Borderline", "source": "ch1", "words": 1, "promoted": True, "clip_id": "a",
    })
    snap = tally.snapshot("w0", "running", 1, 10)
    assert snap["dialects"]["ar_egy"]["promoted_clips"] == 1
    assert worker_alerts({"name": "w0", "state": "running", "updated_ts": time.time() - 200, "samples": []})
    assert promotion_alerts({"ar_egy": {"hours": 2, "promoted_hours": 0.1}})
    assert not promotion_alerts({"ar_egy": {"hours": 0.4, "promoted_hours": 0}})
    assert spend_alert(13, 10)
    assert not spend_alert(11, 10)
    assert not spend_alert(5, 0)
    print("plan12_cohere ok")


if __name__ == "__main__":
    _selftest()
