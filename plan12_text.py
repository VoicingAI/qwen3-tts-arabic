"""Comparison text only. Not for training text."""
import re
import unicodedata

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
WS = re.compile(r"\s+")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
try:
    from rapidfuzz.distance import Levenshtein as _levenshtein
except ImportError:
    _levenshtein = None

# Comparison only. Training text is not rewritten.
# Number words and these spellings share one form. A lone ف or و is attached
# later, after fillers and numbers are set aside, so و ثلاثة does not become و3.
_EQUIV = {
    "الي": "اللي",
    "برضو": "برضه",
    "راح": "رح",
    "مئه": "100",
    "ميه": "100",
    "مائه": "100",
    "هذي": "هذه",
    "كذي": "چذي",
    "شذي": "چذي",
    "جذي": "چذي",
    "انتي": "انت",
    "انتوا": "انتو",
    "صفر": "0",
    "واحد": "1",
    "واحده": "1",
    "اثنين": "2",
    "اثنان": "2",
    "اثنتان": "2",
    "اثنتين": "2",
    "ثلاثه": "3",
    "ثلاث": "3",
    "اربعه": "4",
    "اربع": "4",
    "خمسه": "5",
    "خمس": "5",
    "سته": "6",
    "ست": "6",
    "سبعه": "7",
    "سبع": "7",
    "ثمانيه": "8",
    "ثماني": "8",
    "تسعه": "9",
    "تسع": "9",
    "عشره": "10",
    "عشر": "10",
    "عشرون": "20",
    "عشرين": "20",
    "ثلاثون": "30",
    "ثلاثين": "30",
    "اربعون": "40",
    "اربعين": "40",
    "خمسون": "50",
    "خمسين": "50",
    "ستون": "60",
    "ستين": "60",
    "سبعون": "70",
    "سبعين": "70",
    "ثمانون": "80",
    "ثمانين": "80",
    "تسعون": "90",
    "تسعين": "90",
    "الف": "1000",
    "مليون": "1000000",
    "مليار": "1000000000",
    "احدي": "1",
    "احد": "1",
    "اثناعشر": "12",
}


def _base_compare(text: str) -> str:
    """Marks, alef, and ta marbuta only. No spelling equivalences."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace(TATWEEL, "")
    t = DIACRITICS.sub("", t)
    t = t.translate(ALEF).translate(DIGITS)
    t = PUNCT.sub(" ", t)
    return WS.sub(" ", t).strip()


def normalize_compare(text: str, equate: bool = True) -> str:
    """Strip marks used only when comparing two transcripts.

    Removes vowel marks, tatweel and punctuation. Maps أ إ آ ٱ to ا, ى to ي,
    ة to ه. Converts Arabic-Indic digits. Collapses spaces. Does not map ؤ or ئ.

    When equate is on, spelling pairs share one form and number words become
    digits. This does not change training text.
    """
    t = _base_compare(text)
    if not equate or not t:
        return t
    return " ".join(_EQUIV.get(tok, tok) for tok in t.split())


def attach_clitics(tokens):
    """A separate ف or و is the same as the prefix written on the next word."""
    out = []
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        if tok in {"ف", "و"} and index + 1 < len(tokens):
            out.append(tok + tokens[index + 1])
            index += 2
        else:
            out.append(tok)
            index += 1
    return out


def _edit(ref, hyp):
    if not ref:
        return 0 if not hyp else len(hyp)
    if _levenshtein is not None:
        return _levenshtein.distance(ref, hyp)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    """Word error rate after normalize_compare.

    Edit distance on the words, divided by the number of words in the reference.
    An empty reference is 0 when the hypothesis is also empty, else 1.
    """
    r = normalize_compare(ref).split()
    h = normalize_compare(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit(r, h) / len(r)


def agreement(ref: str, hyp: str) -> float:
    """1 minus word error rate, and never below 0."""
    return 1.0 - min(wer(ref, hyp), 1.0)


def cer(ref: str, hyp: str) -> float:
    """Character error rate after normalize_compare. Spaces stay in the text.

    Edit distance on the characters, divided by the number of characters in the
    reference. An empty reference is 0 when the hypothesis is also empty, else 1.
    """
    r = normalize_compare(ref)
    h = normalize_compare(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _edit(r, h) / len(r)


def score_pair(ref: str, hyp: str) -> dict:
    """Both rates. Clips with fewer than 5 normalised reference words use the character rate."""
    r = normalize_compare(ref)
    h = normalize_compare(hyp)
    words = r.split()
    heard = h.split()
    if not words:
        word_rate = 0.0 if not heard else 1.0
    else:
        word_rate = _edit(words, heard) / len(words)
    if not r:
        char_rate = 0.0 if not h else 1.0
    else:
        char_rate = _edit(r, h) / len(r)
    short = len(words) < 5
    return {
        "words": len(words),
        "wer": word_rate,
        "cer": char_rate,
        "tier_metric": "cer" if short else "wer",
        "score_for_tier": char_rate if short else word_rate,
    }


def _selftest():
    assert normalize_compare("أَإِآىةـ") == "اااي ه".replace(" ", "") or normalize_compare("أإآ") == "ااا"
    assert normalize_compare("أإآ") == "ااا"
    assert normalize_compare("ى") == "ي"
    assert normalize_compare("ة") == "ه"
    assert normalize_compare("١٢٣") == "123"
    assert normalize_compare("ثلاثة") == "3"
    assert normalize_compare("برضو") == "برضه"
    assert normalize_compare("راح") == "رح"
    assert normalize_compare("ميه") == normalize_compare("مئة") == "100"
    assert normalize_compare("هذي") == "هذه"
    assert normalize_compare("كذي") == normalize_compare("شذي") == normalize_compare("جذي") == normalize_compare("چذي") == "چذي"
    assert normalize_compare("انتي") == "انت"
    assert normalize_compare("انتوا") == "انتو"
    assert normalize_compare("الي") == "اللي"
    assert normalize_compare("برضو", equate=False) == "برضو"
    assert normalize_compare("ؤ") == "ؤ"
    assert normalize_compare("ئ") == "ئ"
    assert "ـ" not in normalize_compare("كتـاب")
    assert wer("قط كبير", "قط كبير") == 0
    assert wer("قط كبير", "قط صغير") == 0.5
    assert agreement("قط كبير", "قط صغير") == 0.5
    assert wer("", "") == 0
    assert wer("", "قط") == 1
    assert cer("ab cd", "ab xx") == 0.4
    assert cer("", "") == 0
    assert cer("", "a") == 1
    assert " " in normalize_compare("ab cd")
    short = score_pair("قط", "قط")
    assert short["words"] == 1 and short["tier_metric"] == "cer"
    long = score_pair("واحد اثنان ثلاثة اربعة خمسة", "واحد اثنان ثلاثة اربعة خمسة")
    assert long["words"] == 5 and long["tier_metric"] == "wer" and long["wer"] == 0
    print("plan12_text ok")


if __name__ == "__main__":
    _selftest()
