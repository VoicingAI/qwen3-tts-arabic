"""Screening quiz for Round 3b. Audio only. The transcript stays off the page."""
import hashlib
import random
import statistics
import os as _qwen_os
_DATA = _qwen_os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"

TAG = "internal_11labs"
PASS_REGION = 17
PASS_OWN = 3
CORE_MIN = 2
REPEAT_WARN = 2
ANSWER_SECONDS = 25
GULF = frozenset({"emirati", "kuwaiti", "najdi", "hijazi"})
CORE = ("egyptian", "levantine", "msa")

PICKS = [
    ("0UnHRVB-BSc_0003", "egyptian"),
    ("0fBuqeVquYk_0027", "egyptian"),
    ("-D6nSeqiyUM_0028", "egyptian"),
    ("OFlBXnbUBXg_0090", "kuwaiti"),
    ("WcHlXBCTt6I_0068", "kuwaiti"),
    ("c5pULnSTwYU_0066", "kuwaiti"),
    ("OZmE1TxSdr0_0083", "najdi"),
    ("OUbvdhyY-aM_0193", "najdi"),
    ("5kRBlYBcii4_0226", "najdi"),
    ("01z9ehXOSms_0070", "hijazi"),
    ("lFPtL_60GEU_0122", "hijazi"),
    ("Kz7N_TXYn5Y_0113", "hijazi"),
    ("ScKhTT1DM6E_0355", "emirati"),
    ("1Bw-39A2khw_0219", "emirati"),
    ("MPiGXSWuK1E_0082", "emirati"),
    ("1tPDrJKdOaU_0355", "levantine"),
    ("5tjTFv1BWVU_0035", "levantine"),
    ("59y8Vyh1VDw_0083", "levantine"),
    ("-4bAejn9bjQ_0258", "msa"),
    ("-MapIfwm-BI_0083", "msa"),
    ("-ZjDsB55F-M_0002", "msa"),
]

FOLDER = {
    "egyptian": "eg",
    "kuwaiti": "kuwaiti",
    "najdi": "sa-najdi",
    "hijazi": "sa-hijazi",
    "emirati": "emirati",
    "levantine": "levantine",
    "msa": "msa",
}

DIALECTS = [
    ("egyptian", "مصري", "Egyptian"),
    ("kuwaiti", "كويتي", "Kuwaiti"),
    ("najdi", "سعودي نجدي", "Saudi Najdi"),
    ("hijazi", "سعودي حجازي", "Saudi Hijazi"),
    ("emirati", "إماراتي", "Emirati"),
    ("levantine", "شامي", "Levantine"),
    ("msa", "فصحى", "MSA"),
]
CHOICES = DIALECTS + [
    ("gulf", "خليجي آخر", "other Gulf"),
    ("unsure", "غير متأكد", "not sure"),
]
DIALECT_IDS = {key for key, _ar, _en in DIALECTS}
CHOICE_IDS = {key for key, _ar, _en in CHOICES}

PEOPLE = [
    {"id": "saif", "name": "Saif", "ask": "emirati"},
    {"id": "ahmed", "name": "Ahmed El Dawey", "ask": "emirati"},
    {"id": "yousef", "name": "Yousef", "ask": "kuwaiti"},
    {"id": "abdulaziz", "name": "Abdulaziz", "ask": "najdi"},
    {"id": "leena", "name": "Leena", "ask": "hijazi"},
    {"id": "amr", "name": "Amr", "ask": "egyptian"},
    {"id": "farouq", "name": "Farouq", "ask": "levantine"},
    {"id": "ali", "name": "Ali", "ask": "levantine"},
    {"id": "omar", "name": "Omar", "ask": "levantine"},
    {"id": "test", "name": "TEST", "ask": "emirati", "exclude": True},
]

DIALECT_NAME = {key: label for key, _ar, label in DIALECTS}
V1_ONLY = frozenset({"OGZQSN4xEQg_0103", "-9QCqIA0eyg_0494"})
V2_ONLY = frozenset({"OZmE1TxSdr0_0083", "lFPtL_60GEU_0122"})
RETIRED = [("OGZQSN4xEQg_0103", "najdi"), ("-9QCqIA0eyg_0494", "hijazi")]


def clip_set(doc):
    """v1 heard the original Najdi and Hijazi clips. v2 hears the replacements."""
    ids = {item["clip_id"] for item in doc.get("items", [])}
    if ids & V2_ONLY:
        return "v2"
    if ids & V1_ONLY:
        return "v1"
    return ""


def clip_version(clip_id):
    if clip_id in V2_ONLY:
        return "v2"
    if clip_id in V1_ONLY:
        return "v1"
    return "both"


ANSWER_NAME = {key: label for key, _ar, label in CHOICES}


def region_match(dialect, pick):
    """Gulf clips accept any Gulf answer, including other Gulf. The rest must be exact."""
    if dialect in GULF:
        return pick in GULF or pick == "gulf"
    return pick == dialect


def source_path(clip_id, dialect):
    return f"{_DATA}/yt/yt_clean_24k/{FOLDER[dialect]}/{clip_id}.flac"


def public_id(clip_id):
    """Opaque file name. The YouTube id never appears in the page URL."""
    return hashlib.sha256(("r3b:" + clip_id).encode()).hexdigest()[:16]


def queue(clips, seed):
    """21 clips plus 3 hidden repeats. A repeat sits at least 6 places later."""
    rng = random.Random(seed)
    order = list(clips)
    rng.shuffle(order)
    items = [
        {
            "clip_id": row["clip_id"],
            "public_id": public_id(row["clip_id"]),
            "dialect": row["dialect"],
            "rep": False,
        }
        for row in order
    ]
    shift = 0
    for origin, at in ((0, 8), (1, 14), (2, 20)):
        items.insert(at + shift, {**items[origin], "rep": True})
        shift += 1
    return items


def quiz_minutes(rows):
    """Audio plus 25 seconds a clip, counting the three longest as the repeats."""
    durs = sorted((row["duration_s"] for row in rows), reverse=True)
    audio = sum(durs) + sum(durs[:3])
    return (audio + ANSWER_SECONDS * 24) / 60


def clip_view(item, index, total, nxt):
    """What the browser is allowed to see. No transcript and no answer."""
    return {
        "index": index,
        "clip_no": index + 1,
        "clip_total": total,
        "src": "/clips/" + item["public_id"],
        "next_src": nxt,
    }


def _first(doc):
    return [(item, ans) for item, ans in zip(doc["items"], doc["ans"]) if not item["rep"]]


def _count(rows, predicate):
    return sum(1 for item, ans in rows if predicate(item, ans))


def grade(doc):
    """Score finished or partial answers. A pass needs the full 21 first hearings."""
    pairs = list(zip(doc["items"], doc["ans"]))
    first = _first(doc)
    named = doc.get("dialect") or ""
    exact = _count(first, lambda item, ans: ans["pick"] == item["dialect"])
    region = _count(first, lambda item, ans: region_match(item["dialect"], ans["pick"]))
    own_rows = [(item, ans) for item, ans in first if item["dialect"] == named]
    own_ok = _count(own_rows, lambda item, ans: ans["pick"] == item["dialect"])
    groups = {}
    for dialect in CORE:
        rows = [(item, ans) for item, ans in first if item["dialect"] == dialect]
        groups[dialect] = _count(rows, lambda item, ans: ans["pick"] == dialect)
        groups[dialect + "_n"] = len(rows)
    gulf_rows = [(item, ans) for item, ans in first if item["dialect"] in GULF]
    gulf_exact = _count(gulf_rows, lambda item, ans: ans["pick"] == item["dialect"])
    heard = {}
    changes = 0
    for item, ans in pairs:
        if item["rep"]:
            if heard.get(item["clip_id"]) != ans["pick"]:
                changes += 1
        else:
            heard[item["clip_id"]] = ans["pick"]
    seconds = [ans["seconds"] for _item, ans in pairs if isinstance(ans.get("seconds"), (int, float))]
    core_ok = all(groups[dialect] >= CORE_MIN and groups[dialect + "_n"] == PASS_OWN for dialect in CORE)
    return {
        "correct": exact,
        "region": region,
        "n": len(first),
        "own_ok": own_ok,
        "own_n": len(own_rows),
        "egyptian": groups["egyptian"],
        "egyptian_n": groups["egyptian_n"],
        "levantine": groups["levantine"],
        "levantine_n": groups["levantine_n"],
        "msa": groups["msa"],
        "msa_n": groups["msa_n"],
        "gulf_exact": gulf_exact if named in GULF else None,
        "gulf_n": len(gulf_rows),
        "changes": changes,
        "warn": changes >= REPEAT_WARN,
        "median_s": round(statistics.median(seconds), 1) if seconds else None,
        "passed": (
            len(first) == 21
            and region >= PASS_REGION
            and own_ok == PASS_OWN
            and len(own_rows) == PASS_OWN
            and core_ok
        ),
    }


def first_pick(doc, clip_id):
    for item, ans in _first(doc):
        if item["clip_id"] == clip_id:
            return ans.get("pick")
    return None


def clip_report(docs, clips):
    """One row per clip. Repeats are not a second vote."""
    rows = []
    for clip in clips:
        heard = exact_n = region_n = 0
        answers = []
        for doc in docs:
            pick = first_pick(doc, clip["clip_id"])
            if pick is None:
                answers.append({"name": doc.get("name", ""), "pick": None})
                continue
            heard += 1
            exact_n += pick == clip["dialect"]
            region_n += region_match(clip["dialect"], pick)
            answers.append({"name": doc.get("name", ""), "pick": pick})
        rows.append({
            "clip_id": clip["clip_id"],
            "dialect": clip["dialect"],
            "set": clip.get("set") or clip_version(clip["clip_id"]),
            "answers": answers,
            "heard": heard,
            "exact_pct": round(100 * exact_n / heard) if heard else None,
            "region_pct": round(100 * region_n / heard) if heard else None,
        })
    return rows


def make_codes():
    rng = random.Random(4)
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    used = {
        "hvuen8", "wsxvcw", "a85sju", "hgzsu5", "uspx6e",
        "hxe68t", "p2ay3c", "f39vbk", "3a56js", "w287pz",
    }
    out = []
    for person in PEOPLE:
        while True:
            code = "".join(rng.choice(alphabet) for _ in range(6))
            if code not in used:
                used.add(code)
                break
        out.append({
            "code": code,
            "id": person["id"],
            "name": person["name"],
            "ask": person["ask"],
            "exclude": bool(person.get("exclude")),
        })
    return out
