"""Machine marks for the frozen T1 exam. No generation and no model loads."""
from __future__ import annotations

from plan12_letter import letter_score
from plan12_text import _edit, normalize_compare, wer

PART_A_PRIORITY = ("ar_msa", "ar_egy", "ar_sau", "ar_uae")
PART_B_DIALECTS = (
    "emirati", "kuwaiti", "najdi", "hijazi", "egyptian", "levantine", "msa",
)
JAIS_DIALECTS = {"emirati", "kuwaiti", "najdi", "hijazi", "levantine"}
PART_A_NAMES = {
    "ar_msa": "MSA",
    "ar_egy": "Egyptian",
    "ar_sau": "Saudi",
    "ar_uae": "Emirati",
    "ar_irq": "Iraqi",
    "ar_dza": "Algerian",
    "ar_mar": "Moroccan",
}
HEALTH_THRESHOLDS = {
    "silent_dur_s": 0.3,
    "silent_rms": 0.005,
    "cut_off_cap_fraction": 0.95,
    "too_long_multiple": 2.0,
    "part_a_expected": "reference recording duration",
    "part_b_expected": "non-space characters / 12 seconds",
    "repeating": "a normalised 3-word gram appears at least 3 times in a hypothesis of at least 8 words",
}


def words_of(text: str) -> list[str]:
    return normalize_compare(text or "").split()


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Word alignment. Diagonal is preferred when the costs tie, so a substitution stays a substitution."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    choice = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        choice[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        choice[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            options = (
                (dp[i - 1][j - 1] + sub_cost, "eq" if sub_cost == 0 else "sub"),
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
            )
            best = min(options, key=lambda item: (item[0], ("eq", "sub", "del", "ins").index(item[1])))
            dp[i][j] = best[0]
            choice[i][j] = best[1]
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op = choice[i][j]
        if op in {"eq", "sub"}:
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _stem_match(ref_word: str, hyp_word: str, k: int) -> bool:
    if len(ref_word) < k + 1 or len(hyp_word) < k + 1:
        return False
    ref_stem, hyp_stem = ref_word[:-k], hyp_word[:-k]
    if ref_stem == hyp_stem:
        return True
    return _edit(ref_stem, hyp_stem) <= 1


def ending_error(ref_word: str, hyp_word: str) -> bool:
    """True when some ending length has a matching stem and a different ending."""
    if ref_word == hyp_word:
        return False
    lengths = [1]
    if len(ref_word) >= 3 and len(hyp_word) >= 3:
        lengths.append(2)
    for k in lengths:
        if len(ref_word) <= k or len(hyp_word) <= k:
            continue
        if ref_word[-k:] == hyp_word[-k:]:
            continue
        if _stem_match(ref_word, hyp_word, k):
            return True
    return False


def _rate(count: int, n_ref: int, hyp_empty: bool) -> float:
    if n_ref == 0:
        return 0.0 if hyp_empty else 1.0
    return count / n_ref


def ending_marks(ref: str, hyp: str) -> dict:
    """Shares of reference words. A whole-word substitution is neither mark."""
    ref_words = words_of(ref)
    hyp_words = words_of(hyp)
    if not ref_words:
        empty = 0.0 if not hyp_words else 1.0
        return {"dropped_ending": empty, "dropped_words": empty, "n_ref": 0}
    ending = deleted = 0
    for op, ref_word, hyp_word in align_words(ref_words, hyp_words):
        if op == "del":
            deleted += 1
        elif op == "sub" and hyp_word is not None and ending_error(ref_word, hyp_word):
            ending += 1
    n_ref = len(ref_words)
    return {
        "dropped_ending": ending / n_ref,
        "dropped_words": deleted / n_ref,
        "n_ref": n_ref,
    }


def expected_seconds(text: str, ref_dur: float | None, part: str) -> float:
    if part == "A":
        return float(ref_dur or 0)
    return len((text or "").replace(" ", "")) / 12.0


def health(dur: float, rms: float, ref_dur: float | None, text: str, hyp: str, part: str) -> dict:
    from plan13_eval import _repeating, cap_seconds

    dur = float(dur or 0)
    rms = float(rms or 0)
    cap = cap_seconds(text)
    expected = expected_seconds(text, ref_dur, part)
    flags = {
        "silent": dur < HEALTH_THRESHOLDS["silent_dur_s"] or rms < HEALTH_THRESHOLDS["silent_rms"],
        "cut_off": dur >= HEALTH_THRESHOLDS["cut_off_cap_fraction"] * cap,
        "repeating": bool(_repeating(hyp)) if hyp else False,
        "too_long": expected > 0 and dur > HEALTH_THRESHOLDS["too_long_multiple"] * expected,
    }
    return {
        "flags": flags,
        "broken": any(flags.values()),
        "cap_s": cap,
        "expected_s": expected,
        "thresholds": dict(HEALTH_THRESHOLDS),
    }


def clip_wer(cohere_wer: float | None, jais_wer: float | None, dialect: str, part: str) -> float | None:
    if cohere_wer is None:
        return None
    if part == "B" and dialect in JAIS_DIALECTS:
        if jais_wer is None:
            return None
        return (cohere_wer + jais_wer) / 2
    return cohere_wer


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def percentile(values, p: float) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _seed_average(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    clips = []
    for clip_id, group in grouped.items():
        dialect = group[0].get("dialect")
        part = group[0].get("part")
        cohere = _mean([r.get("wer") for r in group])
        jais = _mean([r.get("jais_wer") for r in group])
        clips.append({
            "id": clip_id,
            "dialect": dialect,
            "part": part,
            "wer": clip_wer(cohere, jais, dialect, part),
            "cohere_wer": cohere,
            "jais_wer": jais,
            "ler": _mean([r.get("ler") for r in group]),
            "dropped_ending": _mean([r.get("dropped_ending") for r in group]),
            "dropped_words": _mean([r.get("dropped_words") for r in group]),
            "sim": _mean([r.get("sim") for r in group]),
            "utmos": _mean([r.get("utmos") for r in group]),
            "broken": _mean([1.0 if r.get("broken") else 0.0 for r in group]),
            "n_seeds": len(group),
        })
    return clips


_QUALITY = (
    "wer", "cohere_wer", "jais_wer", "ler", "dropped_ending", "dropped_words",
    "sim", "sim_other", "utmos",
)
_REASONS = ("silent", "cut_off", "too_long", "repeating")


def _usable(row: dict) -> bool:
    return not row.get("broken") and not row.get("error")


def _value(row: dict, key: str):
    if key == "wer":
        return clip_wer(row.get("wer"), row.get("jais_wer"), row.get("dialect") or "", row.get("part") or "")
    if key == "cohere_wer":
        return row.get("wer")
    return row.get(key)


def _by_seed(rows: list[dict]) -> dict:
    found = {}
    for row in rows:
        found.setdefault(row.get("seed"), []).append(row)
    return found


def _spread(rows: list[dict], key: str) -> dict:
    vals = []
    for group in _by_seed(rows).values():
        got = _mean([_value(row, key) for row in group if _usable(row)])
        if got is not None:
            vals.append(got)
    return {
        key: _mean(vals),
        f"{key}_min": min(vals) if vals else None,
        f"{key}_max": max(vals) if vals else None,
    }


def _rate_spread(rows: list[dict], pick) -> tuple:
    vals = []
    for group in _by_seed(rows).values():
        if group:
            vals.append(sum(1 for row in group if pick(row)) / len(group))
    if not vals:
        return None, None, None
    return _mean(vals), min(vals), max(vals)


def _pack_rows(rows: list[dict]) -> dict:
    packed = {
        "n": len(rows),
        "n_scored": sum(1 for row in rows if _usable(row)),
        "n_broken": sum(1 for row in rows if row.get("broken")),
    }
    packed["broken"], packed["broken_min"], packed["broken_max"] = _rate_spread(
        rows, lambda row: bool(row.get("broken")),
    )
    for reason in _REASONS:
        mean, low, high = _rate_spread(
            rows, lambda row, reason=reason: bool((row.get("flags") or {}).get(reason)),
        )
        packed[f"broken_{reason}"] = mean
        packed[f"broken_{reason}_min"] = low
        packed[f"broken_{reason}_max"] = high
    for key in _QUALITY:
        packed.update(_spread(rows, key))
    scored = [_value(row, "wer") for row in rows if _usable(row)]
    packed["wer_p10"] = percentile(scored, 10)
    packed["wer_p50"] = percentile(scored, 50)
    packed["wer_p90"] = percentile(scored, 90)
    return packed


def _headline_spread(by: dict, names: tuple[str, ...]) -> tuple:
    seeds = {row.get("seed") for group in by.values() for row in group}
    vals = []
    for seed in seeds:
        dialect_means = []
        for name in names:
            group = [row for row in by.get(name, []) if row.get("seed") == seed and _usable(row)]
            got = _mean([_value(row, "wer") for row in group])
            if got is not None:
                dialect_means.append(got)
        if dialect_means:
            vals.append(_mean(dialect_means))
    if not vals:
        return None, None, None
    return _mean(vals), min(vals), max(vals)


def aggregate(rows: list[dict], part: str, mode: str | None = None) -> dict:
    """Quality metrics use non-broken clips. Each number is the mean of the per-seed means."""
    chosen = [row for row in rows if row.get("part") == part]
    if mode is not None:
        chosen = [row for row in chosen if (row.get("mode") or "xvec") == mode]
    by = {}
    for row in chosen:
        by.setdefault(row.get("dialect") or "?", []).append(row)
    dialects = {dialect: _pack_rows(group) for dialect, group in sorted(by.items())}
    part_pack = _pack_rows(chosen)
    names = PART_A_PRIORITY if part == "A" else PART_B_DIALECTS
    headline, low, high = _headline_spread(by, names)
    if part == "A":
        part_pack["priority_wer"] = headline
        part_pack["priority_wer_min"] = low
        part_pack["priority_wer_max"] = high
    else:
        part_pack["overall_wer"] = headline
        part_pack["overall_wer_min"] = low
        part_pack["overall_wer_max"] = high
    part_pack["by_dialect"] = dialects
    part_pack["mode"] = mode
    return part_pack


def score_pair(ref: str, hyp: str) -> dict:
    marks = ending_marks(ref, hyp)
    return {
        "wer": wer(ref, hyp),
        "ler": letter_score(ref, hyp)["letter_rate"],
        "dropped_ending": marks["dropped_ending"],
        "dropped_words": marks["dropped_words"],
    }


def _check(name: str, got, want) -> None:
    if isinstance(want, float):
        if got is None or abs(got - want) > 1e-9:
            raise SystemExit(f"{name}: got {got} want {want}")
    elif got != want:
        raise SystemExit(f"{name}: got {got} want {want}")


def _asserts() -> None:
    same = ending_marks("كتاب جديد", "كتاب جديد")
    _check("identical ending", same["dropped_ending"], 0.0)
    _check("identical deleted", same["dropped_words"], 0.0)

    deleted = ending_marks("السلام عليكم", "السلام")
    _check("deleted words", deleted["dropped_words"], 0.5)
    _check("deleted ending", deleted["dropped_ending"], 0.0)

    last = ending_marks("كتاب", "كتام")
    _check("last letter ending", last["dropped_ending"], 1.0)
    _check("last letter deleted", last["dropped_words"], 0.0)

    stem = ending_marks("كتاب", "كتمم")
    _check("stem edit ending", stem["dropped_ending"], 1.0)
    _check("stem edit deleted", stem["dropped_words"], 0.0)
    _check("stem distance", _edit("كتا", "كتم") <= 1, True)

    other = ending_marks("كتاب", "سيارة")
    _check("unrelated ending", other["dropped_ending"], 0.0)
    _check("unrelated deleted", other["dropped_words"], 0.0)
    _check("unrelated wer", wer("كتاب", "سيارة") > 0, True)

    empty = ending_marks("", "")
    _check("empty both", empty["dropped_ending"], 0.0)
    _check("empty hyp", ending_marks("", "كتاب")["dropped_words"], 1.0)

    rows = [
        {"id": "a", "part": "A", "dialect": "ar_msa", "wer": 0.2, "ler": 0.1, "dropped_ending": 0.0,
         "dropped_words": 0.0, "sim": 0.9, "utmos": 3.0, "broken": False, "jais_wer": None},
        {"id": "a", "part": "A", "dialect": "ar_msa", "wer": 0.4, "ler": 0.1, "dropped_ending": 0.0,
         "dropped_words": 0.0, "sim": 0.9, "utmos": 3.0, "broken": False, "jais_wer": None},
        {"id": "b", "part": "A", "dialect": "ar_egy", "wer": 0.5, "ler": 0.2, "dropped_ending": 0.0,
         "dropped_words": 0.0, "sim": 0.8, "utmos": 2.0, "broken": True, "jais_wer": None},
    ]
    summary = aggregate(rows, "A")
    _check("seed mean", summary["by_dialect"]["ar_msa"]["wer"], 0.3)
    _check("broken excluded", summary["by_dialect"]["ar_egy"]["wer"], None)
    _check("broken rate", summary["by_dialect"]["ar_egy"]["broken"], 1.0)
    _check("priority", summary["priority_wer"], 0.3)

    jais_rows = [
        {"id": "e", "part": "B", "dialect": "emirati", "wer": 0.2, "jais_wer": 0.4, "ler": 0.1,
         "dropped_ending": 0.0, "dropped_words": 0.0, "sim": 1, "utmos": 1, "broken": False},
        {"id": "m", "part": "B", "dialect": "msa", "wer": 0.5, "jais_wer": None, "ler": 0.1,
         "dropped_ending": 0.0, "dropped_words": 0.0, "sim": 1, "utmos": 1, "broken": False},
    ]
    part_b = aggregate(jais_rows, "B")
    _check("jais mean", part_b["by_dialect"]["emirati"]["wer"], 0.3)
    _check("msa cohere only", part_b["by_dialect"]["msa"]["wer"], 0.5)
    print("t1_score asserts passed")


if __name__ == "__main__":
    _asserts()
