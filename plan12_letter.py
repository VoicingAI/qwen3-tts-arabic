"""Part C letter score. CPU only. Does not transcribe."""
import re

from plan12_text import attach_clitics, normalize_compare, _edit

LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
_HASH_TAIL = re.compile(r"[.،,!?؟…\"'»«()\[\]{}]+$")
FILLERS = {"اه", "اا", "ها", "امم"}
_NUMBER_SURFACE = (
    "صفر واحد واحدة اثنين اثنان اثنتان اثنتين ثلاثة ثلاث أربعة اربعة خمسة خمس "
    "ستة ست سبعة سبع ثمانية ثماني تسعة تسع عشرة عشر عشرون عشرين ثلاثون ثلاثين "
    "أربعون اربعون اربعين أربعون خمسون خمسين ستون ستين سبعون سبعين ثمانون ثمانين "
    "تسعون تسعين مئة مائة ألف الف مليون مليار احدى إحدى احد اثنا اثناعشر"
)
NUMBERS = {normalize_compare(word, equate=False) for word in _NUMBER_SURFACE.split()}
NUMBERS.discard("")
PRIORITY = {"ar_msa", "ar_egy", "ar_sau", "ar_gulf", "ar_uae"}
NO_GUARD = {"ar_dza", "ar_mar", "ar_tun"}
TIER_ORDER = ("Tier 1", "Borderline", "Tier 2", "Tier 3")


def non_arabic(jais: str) -> bool:
    """True when more than half of Jais's letters are Latin."""
    latin = len(LATIN.findall(jais or ""))
    arabic = len(ARABIC.findall(jais or ""))
    total = latin + arabic
    return total > 0 and (latin / total) > 0.5


def _drop_hash(text: str) -> str:
    kept = []
    for token in (text or "").split():
        bare = _HASH_TAIL.sub("", token)
        if bare.endswith("#") or token.endswith("#"):
            continue
        kept.append(token)
    return " ".join(kept)


def letters_of(text: str, drop_hash: bool = False, equate: bool = True) -> str:
    """Extra normalisation, then the letters with spaces removed."""
    raw = _drop_hash(text) if drop_hash else (text or "")
    kept = []
    for token in normalize_compare(raw, equate=equate).split():
        if token in FILLERS or token in NUMBERS or token.isdigit():
            continue
        if LATIN.search(token):
            continue
        token = token.replace("ؤ", "ء").replace("ئ", "ء")
        if token == "الي":
            token = "اللي"
        if token:
            kept.append(token)
    if equate:
        kept = attach_clitics(kept)
    return "".join(kept)


def letter_score(jais: str, audar: str, equate: bool = True) -> dict:
    """Letter error rate of Audar against Jais. non_arabic forces Tier 3 later."""
    ref = letters_of(jais, drop_hash=False, equate=equate)
    hyp = letters_of(audar, drop_hash=True, equate=equate)
    flagged = non_arabic(jais)
    if not ref:
        rate = 0.0 if not hyp else 1.0
    else:
        rate = _edit(ref, hyp) / len(ref)
    return {
        "letter_rate": rate,
        "non_arabic": flagged,
        "tier": assign_tier(rate, flagged),
    }


def assign_tier(rate: float, flagged: bool) -> str:
    if flagged or rate > 0.25:
        return "Tier 3"
    if rate <= 0.03:
        return "Tier 1"
    if rate <= 0.10:
        return "Borderline"
    return "Tier 2"


def apply_guard(rows):
    """Tier 3 stays at most 15% of a priority dialect's hours.

    The mildest score-based Tier 3 clips move to Tier 2. Clips flagged
    non_arabic stay in Tier 3. Algerian, Moroccan, and Tunisian are not capped.
    """
    by_dialect = {}
    for row in rows:
        by_dialect.setdefault(row.get("dialect") or "(none)", []).append(row)
    notes = {}
    for dialect, group in by_dialect.items():
        if dialect not in PRIORITY:
            continue
        total = sum(float(row.get("duration") or 0) for row in group)
        cap = 0.15 * total

        def tier3_hours():
            return sum(float(row.get("duration") or 0) for row in group if row.get("tier") == "Tier 3")

        movable = [row for row in group if row.get("tier") == "Tier 3" and not row.get("non_arabic")]
        movable.sort(key=lambda row: (float(row.get("letter_rate") or 0), row.get("clip_id") or ""))
        moved = 0
        for row in movable:
            if tier3_hours() <= cap + 1e-6:
                break
            row["tier"] = "Tier 2"
            row["guard_moved"] = True
            moved += 1
        notes[dialect] = {
            "hours": round(total / 3600, 2),
            "cap_hours": round(cap / 3600, 2),
            "tier3_hours": round(tier3_hours() / 3600, 2),
            "moved": moved,
            "over_cap": tier3_hours() > cap + 1e-6,
        }
    return notes


def _selftest():
    assert letters_of("أهلا آه") != ""
    assert "اه" not in letters_of("قال اه نعم").split() or True
    assert letters_of("قال اه نعم") == letters_of("قال نعم")
    assert letters_of("قال آه نعم") == letters_of("قال نعم")
    assert letters_of("ها امم أأ سلام") == letters_of("سلام")
    assert letters_of("عنده 3 كتب") == letters_of("عنده كتب")
    assert letters_of("عنده ثلاثة كتب") == letters_of("عنده كتب")
    assert letters_of("الي بيت") == "الليبيت"
    assert letters_of("ف بيت") == letters_of("فبيت")
    assert letters_of("و بيت") == letters_of("وبيت")
    assert letters_of("برضو") == letters_of("برضه")
    assert letters_of("راح يجي") == letters_of("رح يجي")
    assert letters_of("ميه") == letters_of("مئة") == ""
    assert letters_of("هذي") == letters_of("هذه")
    assert letters_of("كذي") == letters_of("چذي") == letters_of("شذي") == letters_of("جذي")
    assert letters_of("انت") == letters_of("انتي")
    assert letters_of("انتو") == letters_of("انتوا")
    assert letters_of("3") == letters_of("ثلاثة") == ""
    assert letters_of("برضو", equate=False) != letters_of("برضه", equate=False)
    assert "ؤ" not in letters_of("مؤمن") and "ئ" not in letters_of("مسائل")
    assert letters_of("hello سلام") == letters_of("سلام")
    assert letters_of("keep", drop_hash=False) == ""
    assert letters_of("سلام# مرحبا", drop_hash=True) == letters_of("مرحبا")
    assert letters_of("سلام مرحبا") == "سلاممرحبا"
    assert non_arabic("hello world يا")
    assert not non_arabic("مرحبا hello")
    assert assign_tier(0.03, False) == "Tier 1"
    assert assign_tier(0.0301, False) == "Borderline"
    assert assign_tier(0.10, False) == "Borderline"
    assert assign_tier(0.1001, False) == "Tier 2"
    assert assign_tier(0.25, False) == "Tier 2"
    assert assign_tier(0.2501, False) == "Tier 3"
    assert assign_tier(0.0, True) == "Tier 3"
    rows = [
        {"dialect": "ar_egy", "duration": 100, "letter_rate": 0.5, "tier": "Tier 3", "non_arabic": False, "clip_id": "a"},
        {"dialect": "ar_egy", "duration": 100, "letter_rate": 0.4, "tier": "Tier 3", "non_arabic": False, "clip_id": "b"},
        {"dialect": "ar_egy", "duration": 800, "letter_rate": 0.01, "tier": "Tier 1", "non_arabic": False, "clip_id": "c"},
        {"dialect": "ar_mar", "duration": 100, "letter_rate": 0.9, "tier": "Tier 3", "non_arabic": False, "clip_id": "d"},
    ]
    notes = apply_guard(rows)
    assert rows[0]["tier"] == "Tier 3" and rows[1]["tier"] == "Tier 2"
    assert rows[3]["tier"] == "Tier 3"
    assert notes["ar_egy"]["moved"] == 1
    print("plan12_letter ok")


if __name__ == "__main__":
    _selftest()
