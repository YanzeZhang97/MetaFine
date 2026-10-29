"""Instruction -> task variant.

Order: the exact official strings (eval/configs/*.yaml), then task-specific rules, then
the task's default variant (the Perception variant, so a parse failure never costs the
Perception metric). The rules cover paraphrases:
- synonyms and word forms ("lid", "leftwards", "bluish");
- curly quotes and other Unicode punctuation;
- negation within a clause ("not the cap", "except the blue one");
- for T4, which colour is bound to the left-hand box;
- for T5, letters named without the word "letter", or only by peg colour.
``parse_variant`` and ``parse_instruction`` never raise.
"""
from __future__ import annotations

import re
import unicodedata

DEFAULT_VARIANT = {
    "grasp_part": "cap",
    "grasp_move_mug": "left",
    "toggle_switch_table": "red",
    "put_blocks_into_boxes": "green",
    "insert_letter": "C",
}

_EXACT = {  # eval/configs/*.yaml
    "grasp_part": {"Grasp the cap of the bottle": "cap", "Grasp the body of the bottle": "body"},
    "grasp_move_mug": {"Grasp the mug by the handle and move it to the left": "left",
                       "Grasp the mug by the handle and move it to the right": "right",
                       "Grasp the mug by the handle and move it forward": "forward"},
    "toggle_switch_table": {"Toggle the red switch on the table": "red",
                            "Toggle the blue switch on the table": "blue"},
    "put_blocks_into_boxes": {f"Place the {c} cube in the left-hand box and the remaining cubes in the right-hand box.": c
                              for c in ("red", "green", "blue")},
    "insert_letter": {f"Insert the letter {v} into its slot on the board": v for v in ("C", "o", "R", "L")},
}

# A negation word up to five words before a mention, in the same clause, cancels it.
_NEG = re.compile(r"(?:\b(?:not|never|no|except|other than|instead of|rather than|avoid|avoiding|without|but not|"
                  r"apart from|dont|isnt|doesnt|arent|wont|shouldnt|mustnt)\b|\b\w+n't\b)(?:\W+\w+){0,4}\W*$")
_CLAUSE_END = re.compile(r"[;,.:!?()]|\bbut\b|\bthen\b")


def _norm(text) -> str:
    t = unicodedata.normalize("NFKC", str(text or ""))
    for a, b in (("‘", "'"), ("’", "'"), ("‛", "'"), ("“", '"'), ("”", '"'), ("„", '"'), ("`", "'"),
                 ("–", "-"), ("—", "-"), ("‐", "-")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def _negated(low: str, pos: int) -> bool:
    before = low[:pos]
    cut = [m.end() for m in _CLAUSE_END.finditer(before)]
    return bool(_NEG.search(before[cut[-1] if cut else 0:]))


def _mentions(low: str, table) -> list:
    """Sorted [(pos, variant, negated)] for every match of table = [(variant, regex)]."""
    out = []
    for variant, rx in table:
        for m in re.finditer(rx, low):
            out.append((m.start(), variant, _negated(low, m.start())))
    return sorted(out)


def _pick(hits, variants):
    """First non-negated mention; if every mention is negated and there are two variants,
    the one that was not negated."""
    for _, v, neg in hits:
        if not neg:
            return v
    if hits and len(variants) == 2:
        named = {v for _, v, _ in hits}
        if len(named) == 1:
            return next(x for x in variants if x not in named)
    return None


# ---- T1 ------------------------------------------------------------------------------
_PART = [("cap", r"\b(?:caps?|lids?|tops?|stopper|cork|closures?|neck|head|upper (?:part|end|portion))\b"),
         ("body", r"\b(?:body|bodies|sides?|middle|label|base|bottom|torso|trunk|barrel|belly|cylind\w*|"
                  r"lower (?:part|end|portion)|main part|widest|thickest|wide part|cent(?:er|re)|midsection|mid-section|waist)\b")]


def _grasp_part(low):
    low = re.sub(r"\bon top of\b", " on ", low)
    low = re.sub(r"\b(?:below|under|beneath|underneath)\s+(?:the |its )?(?:caps?|lids?|tops?)\b", " body ", low)
    return _pick(_mentions(low, _PART), ("cap", "body"))


# ---- T2 ------------------------------------------------------------------------------
_DIR = [("left", r"\bleft(?:wards?)?\b"), ("right", r"\bright(?:wards?)?\b(?!-hand)"),
        ("forward", r"\b(?:forwards?|front|ahead|away|farther|further|outwards?)\b|\btowards? the camera\b")]
_MOTION = re.compile(r"\b(?:move|moving|moved|shift|slide|push|pull|carry|translate|bring|drag|transport|displace|"
                     r"put|place|take|relocate|nudge)\w*\b")


def _grasp_move_mug(low):
    hits = _mentions(low, _DIR)
    for mv in _MOTION.finditer(low):  # the direction that follows the motion verb
        got = _pick([h for h in hits if h[0] > mv.end()], ())
        if got:
            return got
    return _pick(hits, ())


# ---- T3 ------------------------------------------------------------------------------
_HOUSING = r"(?!(?:\W+\w+)?\W+(?:base|body|box|housing|device|panel|case|casing|board|frame|block)\b)"
_SWITCH = [("red", rf"\bred(?:dish)?\b{_HOUSING}"), ("blue", rf"\b(?:light-?\s?)?blu(?:e|ish)\b{_HOUSING}")]


def _toggle_switch(low):
    return _pick(_mentions(low, _SWITCH), ("red", "blue"))


# ---- T4 ------------------------------------------------------------------------------
_CUBE = [("red", r"\bred(?:dish)?\b"), ("blue", r"\bblu(?:e|ish)\b"), ("green", r"\bgreen(?:ish)?\b")]
_LEFT = re.compile(r"\bleft\b")


def _left_colour(segment):
    if not _LEFT.search(segment):
        return None
    named = {v for _, v, neg in _mentions(segment, _CUBE) if not neg}
    return named.pop() if len(named) == 1 else None


def _put_blocks(low):
    # The special cube is the one bound to the left-hand box: look for a sentence, then a
    # clause, that mentions "left" and exactly one colour.
    sentences = re.split(r"[;.!?]", low)
    for s in sentences:
        got = _left_colour(s)
        if got:
            return got
    for s in sentences:
        for clause in re.split(r"[,:()]|\band\b|\bthen\b|\bwhile\b|\bwhereas\b|\bbut\b", s):
            got = _left_colour(clause)
            if got:
                return got
    m = re.search(r"\b(?:except|but|other than|apart from)\s+(?:for\s+)?(?:the\s+)?(red|blue|green)\b", low)
    if m and re.search(r"\bright\b", low):  # "everything except the blue one goes right"
        return m.group(1)
    return _pick(_mentions(low, _CUBE), ())


# ---- T5 ------------------------------------------------------------------------------
_LETTER_OF = {"c": "C", "o": "o", "r": "R", "l": "L"}
_PEG_COLOUR = [("C", r"\bred\b"), ("o", r"\bblue\b"), ("R", r"\byellow\b"), ("L", r"\bgreen\b")]
_LETTER_PATTERNS = [
    r"\bletters?[\s\-]*['\"(]?\s*([A-Za-z])\b",                                   # letter R, letter-R, letter (R)
    r"['\"(]([A-Za-z])['\")]",                                                     # 'R', "o", (L)
    r"(?<![A-Za-z'])([CcOoRrLl])['\")]?(?=\s*-?\s*(?:shaped|peg|letter|block|piece|glyph|character|slot)\b)",
    r"(?<![A-Za-z'])([CcOoRrLl])(?![A-Za-z'])",                                    # a lone C / O / R / L
]


def _insert_letter(text):
    low = text.lower()
    for pat in _LETTER_PATTERNS:
        for m in re.finditer(pat, text):
            v = _LETTER_OF.get(m.group(1).lower())
            if v and not _negated(low, m.start()):
                return v
    return _pick(_mentions(low, _PEG_COLOUR), ())


_RULES = {"grasp_part": _grasp_part, "grasp_move_mug": _grasp_move_mug, "toggle_switch_table": _toggle_switch,
          "put_blocks_into_boxes": _put_blocks}


def parse_instruction(task_id: str, instruction: str) -> tuple[str, str]:
    """(variant, how). ``how`` is "exact" for an official string, "rule" when a paraphrase rule
    matched, and "default" when nothing matched (or the input was unusable)."""
    default = DEFAULT_VARIANT.get(task_id, "")
    try:
        text = _norm(instruction)
        exact = _EXACT.get(task_id, {})
        if text in exact:
            return exact[text], "exact"
        for s, v in exact.items():
            if text.lower().rstrip(".") == s.lower().rstrip("."):
                return v, "exact"
        if task_id == "insert_letter":
            got = _insert_letter(text)
        else:
            rule = _RULES.get(task_id)
            got = rule(text.lower()) if rule else None
        return (got, "rule") if got else (default, "default")
    except Exception:
        return default, "default"


def parse_variant(task_id: str, instruction: str) -> str:
    return parse_instruction(task_id, instruction)[0]
