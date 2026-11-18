"""Channel floor, approved spelling, and the Part E vote. CPU only.

Does not load a model, start a GPU, or rewrite train_codes.jsonl.
The codes file stays as it is. New training text is written beside it.
"""
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import modal

from plan12_cohere import dialect_label
from plan12_letter import PRIORITY, TIER_ORDER, letter_score
from plan12_text import attach_clitics, normalize_compare
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

app = modal.App("qwen3-arabic-plan12-e")
vol = modal.Volume.from_name("qwen3-arabic-tts-v2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rapidfuzz")
    .add_local_python_source("plan12_text", "plan12_letter", "plan12_cohere")
)

PART = Path(f"{_DATA}/retx/plan12/partD")
OUT = Path(f"{_DATA}/retx/plan12")
TRAIN = Path(f"{_DATA}/tts_stageA/stageA3/train_codes.jsonl")
RANK = {"Tier 3": 0, "Tier 2": 1, "Borderline": 2, "Tier 1": 3}
REFEREE = {"ar_egy", "ar_msa", "ar_sau", "ar-egy", "ar-saudi"}
VOTE_DIALECTS = PRIORITY | {"ar-egy", "ar-saudi"}
MIN_CHANNEL_S = 0.5 * 3600

# Approved training-text rules only. Comparison equivalences are untouched.
RULES_ALL = (
    ("الي", "اللي"),
    ("دا", "ده"),
    ("كدا", "كده"),
    ("برضه", "برضو"),
    ("بردو", "برضو"),
    ("ازي", "ازاي"),
    ("انتوا", "انتو"),
    ("هاذي", "هذي"),
    ("هاذا", "هذا"),
    ("مائه", "مئه"),
)
RULES_GULF = (
    ("كذي", "چذي"),
    ("شذي", "چذي"),
    ("جذي", "چذي"),
)
GULF_DIALECTS = {"ar_gulf", "ar_uae"}
_MARK = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")


def _mark(ch):
    return unicodedata.category(ch) == "Mn" or ch == "ـ"


def _letter(ch):
    return not _mark(ch) and ("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f")


def _skeleton(word):
    return _MARK.sub("", word).replace("ـ", "")


def _edges(word):
    i = 0
    n = len(word)
    while i < n and not _letter(word[i]):
        i += 1
    j = n
    while j > i and not _letter(word[j - 1]):
        j -= 1
    return word[:i], word[i:j], word[j:]


def _letters(word):
    parts = []
    for ch in word:
        if _mark(ch):
            if parts:
                parts[-1][1].append(ch)
        else:
            parts.append([ch, []])
    return parts


def _transfer(core, target):
    """Copy vowel marks onto the new spelling. A mark on a deleted letter is lost."""
    src = _letters(core)
    dst = _letters(target)
    n, m = len(src), len(dst)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if src[i - 1][0] == dst[j - 1][0] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    lost = False
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] and src[i - 1][0] == dst[j - 1][0]:
            dst[j - 1][1] = src[i - 1][1]
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            dst[j - 1][1] = src[i - 1][1]
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            if src[i - 1][1]:
                lost = True
            i -= 1
        else:
            j -= 1
    out = "".join(letter + "".join(marks) for letter, marks in dst)
    return out, lost


def rules_for(dialect):
    rules = list(RULES_ALL)
    if dialect in GULF_DIALECTS:
        rules.extend(RULES_GULF)
    return rules


def spell_text(text, dialect, with_marks=False):
    """Return new text, rule names that fired, and whether a vowel mark was lost."""
    mapping = dict(rules_for(dialect))
    fired = []
    lost = False
    out = []
    for word in (text or "").split(" "):
        if word == "":
            out.append(word)
            continue
        left, core, right = _edges(word)
        skel = _skeleton(core)
        target = mapping.get(skel)
        if not target or skel == target:
            out.append(word)
            continue
        fired.append(f"{skel}→{target}")
        if not with_marks or core == skel:
            out.append(left + target + right)
            continue
        updated, word_lost = _transfer(core, target)
        lost = lost or word_lost
        out.append(left + (target if word_lost else updated) + right)
    return " ".join(out), fired, lost


def spell_row(plain, diac, dialect):
    new_plain, fired, _ = spell_text(plain or "", dialect, with_marks=False)
    no_diac = False
    new_diac = diac or ""
    if diac:
        updated, diac_fired, lost = spell_text(diac, dialect, with_marks=True)
        plain_rules = set(fired)
        if lost or plain_rules - set(diac_fired):
            no_diac = True
            new_diac = new_plain
        else:
            new_diac = updated
    return {
        "text": new_plain,
        "text_original": plain or "",
        "text_diac": new_diac,
        "text_diac_original": diac or "",
        "no_diac": no_diac,
        "rules": fired,
    }


def _pairs(text):
    surfaces = []
    equated = []
    for word in (text or "").split():
        eq = normalize_compare(word, equate=True)
        if not eq:
            continue
        surfaces.append(word)
        equated.append(eq)
    glued = attach_clitics(equated)
    out = []
    index = 0
    for token in glued:
        if (
            index + 1 < len(equated)
            and equated[index] in {"ف", "و"}
            and token == equated[index] + equated[index + 1]
        ):
            out.append(surfaces[index] + surfaces[index + 1])
            index += 2
        else:
            out.append(surfaces[index])
            index += 1
    return out, glued


def _align(ref, hyp):
    n, m = len(ref), len(hyp)
    if n * m > 8000:
        return None
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    ops = []
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("eq", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _place(ops):
    aligned = {}
    inserts = defaultdict(list)
    cursor = -1
    for op, i, j in ops:
        if op == "ins":
            inserts[cursor].append(j)
        else:
            aligned[i] = j
            cursor = i
    return aligned, inserts


def _merge_gaps(a_idx, c_idx, a_eq, c_eq):
    """Pair insertion tokens. Agreeing words become one slot."""
    if _align([a_eq[k] for k in a_idx], [c_eq[k] for k in c_idx]) is None:
        return []
    ops = _align([a_eq[k] for k in a_idx], [c_eq[k] for k in c_idx])
    slots = []
    for op, i, j in ops:
        slots.append((a_idx[i] if i is not None else None, c_idx[j] if j is not None else None))
    return slots


def _slot_choice(dialect, j, a, c):
    """j, a, c are equated words. An empty string means that system has no word here."""
    referee = dialect in REFEREE
    same_ac = bool(a) and a == c
    if j and a == j and c == j:
        return "jais", 1.0
    if j and (a == j or c == j):
        return "jais", 2 / 3
    if not j:
        if same_ac:
            return "cohere", 2 / 3
        return None, None
    if not a and not c:
        if referee:
            return "drop", 2 / 3
        return "jais", 1 / 3
    if same_ac and a != j:
        if referee:
            return "cohere", 2 / 3
        return "jais", 1 / 3
    if referee and c and c != j and a != j:
        return "cohere", 1 / 3
    return "jais", 1 / 3


def vote_clip(dialect, jais, audar, cohere):
    """Word vote. Jais wins a tie. Cohere referees Egyptian, MSA and Saudi."""
    j_s, j_e = _pairs(jais)
    a_s, a_e = _pairs(audar)
    c_s, c_e = _pairs(cohere)
    if not j_e and not a_e and not c_e:
        return {"text": "", "confidence": 0.0, "same_as_jais": True, "words": 0}
    aj = _align(j_e, a_e)
    cj = _align(j_e, c_e)
    if aj is None or cj is None:
        return {"text": " ".join(j_s), "confidence": 0.0, "same_as_jais": True, "words": len(j_e)}
    a_at, a_ins = _place(aj)
    c_at, c_ins = _place(cj)

    def _eq(aligned, eqs, gap):
        idx = aligned.get(gap)
        return "" if idx is None else eqs[idx]

    def _surf(aligned, surfs, gap):
        idx = aligned.get(gap)
        return "" if idx is None else surfs[idx]

    surfaces = []
    chosen = []
    confs = []
    for gap in range(-1, len(j_e)):
        if gap >= 0:
            choice, conf = _slot_choice(dialect, j_e[gap], _eq(a_at, a_e, gap), _eq(c_at, c_e, gap))
            if choice == "jais":
                surfaces.append(j_s[gap])
                chosen.append(j_e[gap])
            elif choice == "cohere":
                surfaces.append(_surf(c_at, c_s, gap))
                chosen.append(_eq(c_at, c_e, gap))
            elif choice == "audar":
                surfaces.append(_surf(a_at, a_s, gap))
                chosen.append(_eq(a_at, a_e, gap))
            if conf is not None:
                confs.append(conf)
        if not a_ins.get(gap) and not c_ins.get(gap):
            continue
        for a_i, c_i in _merge_gaps(a_ins.get(gap, []), c_ins.get(gap, []), a_e, c_e):
            aw = a_e[a_i] if a_i is not None else ""
            cw = c_e[c_i] if c_i is not None else ""
            choice, conf = _slot_choice(dialect, "", aw, cw)
            if choice == "cohere" and c_i is not None:
                surfaces.append(c_s[c_i])
                chosen.append(c_e[c_i])
            elif choice == "audar" and a_i is not None:
                surfaces.append(a_s[a_i])
                chosen.append(a_e[a_i])
            if conf is not None:
                confs.append(conf)
    text = " ".join(s for s in surfaces if s)
    same = chosen == j_e
    conf = sum(confs) / len(confs) if confs else 0.0
    return {"text": text, "confidence": conf, "same_as_jais": same, "words": len(confs)}


def final_tier(current, voted):
    if voted["confidence"] >= 0.8 and voted["same_as_jais"]:
        return current
    if voted["confidence"] >= 0.8:
        return "Tier 2"
    return "Tier 3"


def _tiers(jais, audar, cohere):
    old = letter_score(jais, audar, equate=False)
    new = letter_score(jais, audar, equate=True)
    old_tier = old["tier"]
    new_tier = new["tier"]
    if cohere and letter_score(jais, cohere, equate=True)["letter_rate"] <= 0.03:
        new_tier = "Tier 1"
    if cohere and letter_score(jais, cohere, equate=False)["letter_rate"] <= 0.03:
        old_tier = "Tier 1"
    return old_tier, new_tier


def _hours(rows):
    return round(sum(row[0] for row in rows) / 3600, 2)


def _by_dialect(rows):
    buckets = defaultdict(list)
    for dur, dialect in rows:
        buckets[dialect].append((dur, dialect))
    lines = []
    for dialect, group in sorted(buckets.items(), key=lambda item: -_hours(item[1])):
        lines.append(f"- {dialect}: {_hours(group)} h")
    lines.append(f"- all: {_hours(rows)} h")
    return lines


def _tier_table(rows):
    table = defaultdict(lambda: defaultdict(float))
    for dur, dialect, tier in rows:
        table[dialect][tier] += dur
    lines = [
        "| dialect | hours | Tier 1 | Borderline | Tier 2 | Tier 3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    total = Counter()
    for dialect in sorted(table, key=lambda name: -sum(table[name].values())):
        sec = table[dialect]
        total.update(sec)
        cells = " | ".join(f"{sec[tier] / 3600:.2f}" for tier in TIER_ORDER)
        lines.append(f"| {dialect} | {sum(sec.values()) / 3600:.2f} | {cells} |")
    cells = " | ".join(f"{total[tier] / 3600:.2f}" for tier in TIER_ORDER)
    lines.append(f"| all | {sum(total.values()) / 3600:.2f} | {cells} |")
    return lines


def _bad_channels(channel_s, channel_t3):
    bad = {}
    for source, seconds in channel_s.items():
        if seconds < MIN_CHANNEL_S:
            continue
        share = channel_t3[source] / seconds if seconds else 0
        if share >= 0.30:
            bad[source] = (seconds, channel_t3[source], share)
    return bad


def _keep_ours(dur, words, source, bad):
    if dur < 2 or dur > 20:
        return False
    if dur <= 0 or words / dur < 1:
        return False
    return source not in bad


def _keep_darbuka(dur, words):
    if dur < 2 or dur > 20:
        return False
    if dur <= 0 or words / dur < 1:
        return False
    return True


def _apply_vote(dialect, tier, jais, audar, cohere, spelled):
    if dialect not in VOTE_DIALECTS or tier not in ("Borderline", "Tier 2") or not cohere:
        return tier, spelled, False
    voted = vote_clip(dialect, jais, audar, cohere)
    new_tier = final_tier(tier, voted)
    if voted["same_as_jais"]:
        return new_tier, spelled, False
    text = spell_text(voted["text"], dialect, with_marks=False)[0]
    changed = dict(spelled)
    changed["text"] = text
    changed["no_diac"] = True
    changed["text_diac"] = text
    return new_tier, changed, True


@app.function(
    image=image, volumes={"/data": vol}, cpu=4, memory=32768, timeout=6 * 60 * 60, retries=0,
)
def run():
    vol.reload()
    t0 = time.time()
    cohere = {}
    for path in sorted((PART / "gpu").glob("w*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("sha_ok") is False:
                    continue
                cohere[row["clip_id"]] = row.get("cohere") or ""
    print(f"cohere {len(cohere)} {time.time() - t0:.0f}s", flush=True)

    spell = {}
    rule_counts = Counter()
    diac_lost = 0
    changed_clips = 0
    train_rows = 0
    with TRAIN.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            train_rows += 1
            plain = row.get("text_plain") or row.get("text") or ""
            diac = row.get("text_diac2") or row.get("text_diac") or ""
            dialect = row.get("dialect") or ""
            done = spell_row(plain, diac, dialect)
            if done["rules"]:
                changed_clips += 1
                for rule in set(done["rules"]):
                    rule_counts[rule] += 1
                if done["no_diac"]:
                    diac_lost += 1
                cid = row.get("clip_id") or ""
                done["clip_id"] = cid
                done["dialect"] = dialect
                spell[cid] = done
            if train_rows % 50000 == 0:
                print(f"train {train_rows} changed {changed_clips}", flush=True)
    print(f"train rows {train_rows} changed {changed_clips}", flush=True)

    ours_final = []
    drop = defaultdict(lambda: defaultdict(float))
    move = defaultdict(lambda: Counter())
    channel_s = defaultdict(float)
    channel_t3 = defaultdict(float)
    pending = []
    with (PART / "our_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            dur = float(row.get("duration") or 0)
            dialect = row["dialect"]
            jais = row.get("jais") or ""
            audar = row.get("audar") or ""
            text_c = cohere.get(row["clip_id"], "")
            old, new = _tiers(jais, audar, text_c)
            if RANK[new] > RANK[old]:
                move[dialect]["up"] += dur
                if new == "Tier 1":
                    move[dialect]["to_tier1"] += dur
            elif RANK[new] < RANK[old]:
                move[dialect]["down"] += dur
            source = row.get("source") or "(no channel)"
            channel_s[source] += dur
            if new == "Tier 3":
                channel_t3[source] += dur
            pending.append((dur, dialect, row.get("words") or 0, source, new, jais, audar, text_c, row["clip_id"]))
    bad = _bad_channels(channel_s, channel_t3)
    voted_hours = defaultdict(float)
    changed_vote = defaultdict(float)
    for dur, dialect, words, source, new, jais, audar, text_c, cid in pending:
        reasons = []
        if dur < 2 or dur > 20:
            reasons.append("short or long")
        if dur <= 0 or words / dur < 1:
            reasons.append("under 1 word/second")
        if source in bad:
            reasons.append("channel")
        if reasons:
            for reason in reasons:
                drop[dialect][reason] += dur
            drop[dialect]["any"] += dur
            continue
        spelled = spell.get(cid) or {
            "text": jais, "text_original": jais, "text_diac": "", "text_diac_original": "",
            "no_diac": False, "rules": [],
        }
        tier, spelled, voted_off = _apply_vote(dialect, new, jais, audar, text_c, spelled)
        if dialect in VOTE_DIALECTS and new in ("Borderline", "Tier 2") and text_c:
            voted_hours[dialect] += dur
            if voted_off:
                changed_vote[dialect] += dur
                spelled["clip_id"] = cid
                spelled["dialect"] = dialect
                spelled["vote"] = True
                spell[cid] = spelled
        ours_final.append((dur, dialect, tier))
    print(f"ours {len(pending)} kept {len(ours_final)}", flush=True)

    darbuka_final = []
    d_drop = defaultdict(lambda: defaultdict(float))
    d_move = defaultdict(lambda: Counter())
    labels = {}
    for path in sorted((PART / "darbuka_dialect").glob("[0-9][0-9][0-9][0-9].jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("clip_id") or ""
                if cid:
                    labels[cid] = dialect_label(row.get("dialect_value"))
    seen_d = 0
    for path in sorted(Path(f"{_DATA}/retx/plan12/darbuka_parts").glob("[0-9][0-9][0-9][0-9].jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("_empty"):
                    continue
                jais = row.get("jais") or ""
                audar = row.get("audar") or ""
                text_c = row.get("cohere") or ""
                if not jais and not audar and not text_c:
                    continue
                dialect = labels.get(row.get("clip_id") or "", "unknown")
                dur = float(row.get("duration") or 0)
                old, new = _tiers(jais, audar, text_c)
                if RANK[new] > RANK[old]:
                    d_move[dialect]["up"] += dur
                elif RANK[new] < RANK[old]:
                    d_move[dialect]["down"] += dur
                words = len(normalize_compare(jais, equate=False).split())
                reasons = []
                if dur < 2 or dur > 20:
                    reasons.append("short or long")
                if dur <= 0 or words / dur < 1:
                    reasons.append("under 1 word/second")
                if reasons:
                    for reason in reasons:
                        d_drop[dialect][reason] += dur
                    d_drop[dialect]["any"] += dur
                    continue
                spelled = {
                    "text": jais, "text_original": jais, "text_diac": "", "text_diac_original": "",
                    "no_diac": False, "rules": [],
                }
                tier, _spelled, voted_off = _apply_vote(dialect, new, jais, audar, text_c, spelled)
                if dialect in VOTE_DIALECTS and new in ("Borderline", "Tier 2") and text_c:
                    voted_hours[dialect] += dur
                    if voted_off:
                        changed_vote[dialect] += dur
                darbuka_final.append((dur, dialect, tier))
                seen_d += 1
                if seen_d % 100000 == 0:
                    print(f"darbuka {seen_d}", flush=True)
    print(f"darbuka kept {len(darbuka_final)}", flush=True)

    lines = [
        "# Part E",
        "",
        "CPU only. No model was loaded and no GPU was started. `train_codes.jsonl` was not rewritten.",
        "The new training text is on the changed rows below, with the original kept beside it.",
        "",
        "## Channel drop",
        "",
        "A whole channel is dropped only when it has at least 0.5 h and at least 30% of that time in Tier 3.",
        "Shorter channels stay. Their clips are kept or dropped one by one, on length and speaking rate.",
        "A clip under 2 s or over 20 s is dropped. A clip under 1 Jais word per second is dropped.",
        f"Channels dropped: {len(bad)}.",
        "",
    ]
    for source, (seconds, t3, share) in sorted(bad.items(), key=lambda item: -item[1][0]):
        lines.append(f"- `{source}` {seconds / 3600:.2f} h, Tier 3 {share * 100:.1f}%")
    lines += ["", "### Hours dropped, our data", ""]
    for dialect in sorted(drop, key=lambda name: -drop[name]["any"]):
        item = drop[dialect]
        lines.append(
            f"- {dialect}: {item['any'] / 3600:.2f} h "
            f"(short or long {item['short or long'] / 3600:.2f}, "
            f"under 1 word/second {item['under 1 word/second'] / 3600:.2f}, "
            f"channel {item['channel'] / 3600:.2f})"
        )
    lines.append(f"- all: {sum(v['any'] for v in drop.values()) / 3600:.2f} h")
    lines += ["", "### Hours dropped, Darbuka", "", "No channel cut.", ""]
    for dialect in sorted(d_drop, key=lambda name: -d_drop[name]["any"]):
        item = d_drop[dialect]
        lines.append(f"- {dialect}: {item['any'] / 3600:.2f} h (under 1 word/second {item['under 1 word/second'] / 3600:.2f})")
    lines += [
        "",
        "## Spelling",
        "",
        "Applied only to our training text. Gulf and Emirati rules run on `ar_gulf` and `ar_uae` only.",
        "No other proposal rule was applied.",
        f"Training rows read: {train_rows:,}. Rows whose text changed: {changed_clips:,}.",
        f"Rows with vowel marks that could not be carried onto the new spelling, flagged `no_diac`: {diac_lost:,}.",
        "Each count is clips, and a clip is counted once per rule.",
        "",
    ]
    for src, dst in list(RULES_ALL) + list(RULES_GULF):
        lines.append(f"- {src}→{dst}: {rule_counts[f'{src}→{dst}']:,} clips")
    lines += [
        "",
        "## Vote",
        "",
        "Borderline and Tier 2 clips in the priority dialects are voted word by word.",
        "Jais wins when Audar or Cohere agrees with him, and Jais wins a three-way split outside the referee dialects.",
        "Cohere is the referee for Egyptian, MSA and Saudi, including Darbuka `ar-egy` and `ar-saudi`.",
        "There, a word both others reject is replaced by Cohere's word. On Gulf and Emirati, Jais's word stays.",
        "A clip whose average word confidence is at least 0.8 keeps its tier when the vote still matches Jais, otherwise it becomes Tier 2.",
        "Below 0.8 it becomes Tier 3. Tier 1 is left as it was. Other dialects are not voted.",
        "When the voted text differs from Jais, the row is flagged `no_diac` and the vowel text is the voted text.",
        "",
        "Hours that entered the vote, and hours whose text left Jais:",
        "",
    ]
    for dialect in sorted(voted_hours, key=lambda name: -voted_hours[name]):
        lines.append(f"- {dialect}: voted {voted_hours[dialect] / 3600:.2f} h, text changed {changed_vote[dialect] / 3600:.2f} h")
    lines += ["", "## Final tier table", "", "### Our data", ""]
    lines += _tier_table(ours_final)
    lines += ["", "### Darbuka", ""]
    lines += _tier_table(darbuka_final)
    lines += ["", "### Combined", "", "`ar-egy` is added to `ar_egy`. `ar-saudi` is added to `ar_sau`. `unknown` stays separate.", ""]
    combined = [(dur, "ar_egy" if dialect == "ar-egy" else "ar_sau" if dialect == "ar-saudi" else dialect, tier)
                for dur, dialect, tier in ours_final + darbuka_final]
    lines += _tier_table(combined)
    lines += ["", "Stopped here.", ""]
    text_path = OUT / "train_text.jsonl"
    with text_path.open("w", encoding="utf-8") as handle:
        for row in spell.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines.insert(4, f"Changed training rows, with the original text kept in `text_original`: `{text_path}`.")
    text = "\n".join(lines) + "\n"
    path = OUT / "partE_report.md"
    path.write_text(text, encoding="utf-8")
    (PART / "partE_channels.json").write_text(json.dumps([
        {"source": source, "hours": round(seconds / 3600, 2), "tier3_pct": round(100 * share, 1)}
        for source, (seconds, _t3, share) in sorted(bad.items(), key=lambda item: -item[1][0])
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    vol.commit()
    print(f"wrote {path} {len(text)} chars", flush=True)
    return {"chars": len(text), "channels": len(bad), "changed": changed_clips}


@app.local_entrypoint()
def go():
    call = run.spawn()
    print("CALL", call.object_id, flush=True)


def _selftest():
    row = spell_row("الي دا كدا برضه بردو ازي انتوا هاذي هاذا مائه", "", "ar_egy")
    assert row["text"] == "اللي ده كده برضو برضو ازاي انتو هذي هذا مئه"
    assert row["no_diac"] is False
    gulf = spell_row("كذي شذي جذي كذي", "", "ar_gulf")
    assert gulf["text"] == "چذي چذي چذي چذي"
    held = spell_row("كذي في مصر", "", "ar_egy")
    assert held["text"] == "كذي في مصر"
    marked = spell_row("الي", "اَلِي", "ar_msa")
    assert marked["text"] == "اللي"
    assert marked["no_diac"] is True or "ل" in marked["text_diac"]
    same = vote_clip("ar_egy", "الي بيت", "اللي بيت", "اللي بيت")
    assert same["same_as_jais"] and same["confidence"] == 1
    joined = vote_clip("ar_egy", "و بيت", "وبيت", "وبيت")
    assert joined["same_as_jais"] and joined["confidence"] == 1
    odd = vote_clip("ar_egy", "دا بيت", "ده بيت", "ده بيت")
    assert odd["confidence"] >= 0.8
    split = vote_clip("ar_gulf", "كلمة", "اخرى", "ثالثة")
    assert split["same_as_jais"]
    assert final_tier("Borderline", {"confidence": 0.9, "same_as_jais": True}) == "Borderline"
    assert final_tier("Tier 2", {"confidence": 0.9, "same_as_jais": False}) == "Tier 2"
    assert final_tier("Borderline", {"confidence": 0.5, "same_as_jais": True}) == "Tier 3"
    print("plan12_e ok")


if __name__ == "__main__":
    _selftest()
