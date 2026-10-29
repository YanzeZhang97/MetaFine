"""Dev: synthetic instruction paraphrases for training the learned instruction classifier.

Each task has a small grammar (verbs, synonyms, sentence frames) whose label is fixed by
construction: the part to grasp (T1), the direction of motion (T2), the switch colour (T3),
the cube that goes into the left-hand box (T4), the letter to insert (T5). The frames include
the hard cases a bag of words gets wrong: negation ("not the cap"), distractor mentions (the red
switch body, the handle's side), attribute binding (which colour goes left) and self-corrections.
Surface noise: casing, punctuation, polite prefixes, typos.

The 226 sentences in dev/test_instructions.py are NOT used here; they are the held-out test.

  python dev/gen_instructions.py OUT.jsonl [--per-task 6000]
"""
from __future__ import annotations

import argparse
import json
import random

TASKS = {
    "grasp_part": ["cap", "body"],
    "grasp_move_mug": ["left", "right", "forward"],
    "toggle_switch_table": ["red", "blue"],
    "put_blocks_into_boxes": ["red", "blue", "green"],
    "insert_letter": ["C", "o", "R", "L"],
}

PREFIX = ["", "", "", "", "Please ", "Robot, ", "Now ", "Your task: ", "Task: ", "I want you to ", "You should ",
          "Go ahead and ", "Carefully ", "Next, ", "Okay, ", "Could you ", "Kindly "]


def a_an(word: str) -> str:
    """Article for a word or a single letter's name ("an L", "a C")."""
    vowel = word.upper() in "AEFHILMNORSX" if len(word) == 1 else word[:1].lower() in "aeiou"
    return ("an " if vowel else "a ") + word


# --------------------------------------------------------------------------- T1
T1_PART = {
    "cap": ["cap", "lid", "top", "screw cap", "bottle cap", "closure", "top part", "upper part", "cover", "neck",
            "tip", "top end", "head", "cap on top", "uppermost part", "twist cap"],
    "body": ["body", "main body", "middle", "side", "midsection", "middle part", "cylindrical part", "lower part",
             "base", "bottom", "bottom part", "widest part", "torso", "trunk", "barrel", "label", "belly", "center",
             "central part", "main part", "body part", "lower half", "sides", "waist", "bulk"],
}
T1_POS = {
    "cap": ["at the very top", "near its top", "on top", "at the top", "above the body", "at its upper end",
            "right at the top", "from the top", "by the top end", "where the cap is"],
    "body": ["below the cap", "under the cap", "beneath the cap", "near the bottom", "near its base",
             "in the middle", "halfway down", "around the middle", "around its belly", "at mid-height",
             "lower down", "away from the cap", "by the part below the cap"],
}
T1_VERB = ["grasp", "grab", "pick up", "hold", "take hold of", "grip", "seize", "clutch", "get", "take", "lift",
           "pinch", "secure", "hold onto", "close the gripper on", "close your gripper around",
           "use the gripper to grasp", "reach for and grasp", "catch"]
T1_BOTTLE = ["the bottle", "the bottle", "the bottle", "the plastic bottle", "the bottle on the table", "the water bottle",
             "the pink bottle", "the container"]


def t1(r: random.Random, lab: str) -> str:
    oth = "body" if lab == "cap" else "cap"
    P, O = r.choice(T1_PART[lab]), r.choice(T1_PART[oth])
    v, v2 = r.choice(T1_VERB), r.choice(T1_VERB)
    B = r.choice(T1_BOTTLE)
    Bs = B.replace("the ", "the ", 1) + "'s"
    frames = [
        f"{v} the {P} of {B}", f"{v} {Bs} {P}", f"{v} {B} by its {P}", f"{v} {B} by the {P}",
        f"{v} {B} at its {P}", f"{v} {B} around its {P}", f"{v} the bottle {P}", f"{v} the {P}",
        f"{v} {B}, holding it by the {P}", f"{v} the {P} of {B}, not the {O}", f"{v} {B} by the {P}, not by the {O}",
        f"{v} {Bs} {P} rather than its {O}", f"{v} {Bs} {P} instead of the {O}",
        f"don't {v2} the {O}; {v} the {P}", f"do not {v2} the {O}, {v} the {P} instead",
        f"avoid the {O} and {v} {B} by its {P}", f"not the {O}: {v} the {P}",
        f"the {P} is what you should {v}", f"leave the {O} alone and {v} the {P}",
        f"make sure you {v} the {P} of {B}", f"{v} {B} {r.choice(T1_POS[lab])}",
        f"{v} {B} {r.choice(T1_POS[lab])}, not the {O}", f"your gripper should close on {Bs} {P}",
        f"{v} the {P} of {B} and hold it", f"{v} {B} by grabbing its {P}",
        f"go for the {P} of {B}, not its {O}", f"{v} {B}; use the {P}, not the {O}",
        f"the {O} is off limits, {v} the {P}", f"{v} {B} with the fingers on its {P}",
        f"target the {P} of {B} and {v} it", f"{v} {B} using its {P} as the grasp point",
    ]
    return r.choice(frames)


# --------------------------------------------------------------------------- T2
T2_DIR = {
    "left": ["to the left", "left", "leftward", "leftwards", "to the left side", "toward the left", "towards the left",
             "to your left", "in the leftward direction", "to the left-hand side", "sideways to the left",
             "a bit to the left", "10 cm to the left", "ten centimeters to the left", "over to the left",
             "to the left by ten centimetres", "a little to the left", "left a bit"],
    "right": ["to the right", "right", "rightward", "rightwards", "to the right side", "toward the right",
              "towards the right", "to your right", "in the rightward direction", "to the right-hand side",
              "sideways to the right", "a bit to the right", "10 cm to the right", "ten centimeters to the right",
              "over to the right", "to the right by ten centimetres", "a little to the right", "right a bit"],
    "forward": ["forward", "forwards", "ahead", "to the front", "in the forward direction", "away from you",
                "away from the robot", "farther away", "further from you", "away from the robot base", "straight ahead",
                "toward the camera", "10 cm forward", "forward by ten centimetres", "a little forward", "further out",
                "out in front", "away from its base", "frontward", "forward a bit", "ahead of you"],
}
T2_GRASP = ["grasp the mug by the handle", "grasp the mug by its handle", "grab the mug by the handle",
            "pick up the mug by its handle", "hold the mug by the handle", "take the mug by its handle",
            "grip the mug handle", "grasp the handle of the mug", "grab the mug's handle", "seize the mug by its handle",
            "grasp the cup by the handle", "hold the handle of the cup", "take hold of the mug's handle",
            "grasp the coffee mug by the handle", "pick the mug up by its handle"]
T2_MOVE = ["move it", "move the mug", "slide it", "shift it", "push it", "carry it", "bring it", "drag it",
           "translate it", "take it", "relocate it", "displace it", "slide the mug", "shift the mug"]


def t2(r: random.Random, lab: str) -> str:
    D = r.choice(T2_DIR[lab])
    Do = r.choice(T2_DIR[r.choice([x for x in T2_DIR if x != lab])])
    G, M = r.choice(T2_GRASP), r.choice(T2_MOVE)
    side = r.choice(["left", "right"])
    frames = [
        f"{G} and {M} {D}", f"{G}, then {M} {D}", f"{G}; {M} {D}", f"{G} and {M} {D}.",
        f"{M.replace(' it', ' the mug')} {D} after you {G}", f"holding the handle, {M.replace(' it', ' the mug')} {D}",
        f"while holding its handle, {M.replace(' it', ' the mug')} {D}", f"{G} and {M} {D}, not {Do}",
        f"{G} and {M} {D} rather than {Do}", f"don't move it {Do}; {G} and move it {D}",
        f"{G} (on its {side} side) and {M} {D}", f"{G} on the {side} and {M} {D}",
        f"{M.replace(' it', ' the mug')} {D} while holding its handle", f"using the handle, {M.replace(' it', ' the mug')} {D}",
        f"{G} and {M} {D} by about 10 cm", f"{G}. then {M} {D}", f"{G}, and {M} {D}, not {Do}",
        f"first {G}, then {M} {D}", f"{G} and {M} {D}; do not move it {Do}",
        f"the mug should end up {D} of where it is: {G} and {M} there" if lab != "forward" else f"{G} and {M} {D} of where it is",
    ]
    return r.choice(frames)


# --------------------------------------------------------------------------- T3
T3_COL = {"red": ["red", "reddish", "red-colored", "red-coloured", "crimson", "scarlet"],
          "blue": ["blue", "bluish", "light-blue", "light blue", "blue-colored", "blue-coloured", "sky-blue",
                   "pale blue", "baby blue"]}
T3_OBJ = ["switch", "switch", "slider", "toggle", "toggle switch", "button", "lever", "slide switch", "dip switch"]
T3_VERB = ["toggle", "flip", "push", "slide", "switch", "press", "turn on", "actuate", "move", "activate", "operate",
           "shift", "throw", "flick"]
T3_LOC = ["", "", " on the table", " on the red device", " on the red switch box", " on the red base",
          " on the red body", " of the red switch", " on the device", " on the panel"]


def t3(r: random.Random, lab: str) -> str:
    oth = "blue" if lab == "red" else "red"
    C, O = r.choice(T3_COL[lab]), r.choice(T3_COL[oth])
    S, v = r.choice(T3_OBJ), r.choice(T3_VERB)
    L = r.choice(T3_LOC)
    frames = [
        f"{v} the {C} {S}{L}", f"{v} the {S} that is {C}", f"{v} the {S} colored {C}", f"{v} the {S} whose slider is {C}",
        f"{v} the {S} with the {C} slider", f"{v} the {S} with the {C} button on the red base",
        f"on the red switch box, {v} the {C} {S}", f"{v} the {C} one", f"{v} the {C} {S}, not the {O} one",
        f"do not touch the {O} {S}; {v} the {C} one", f"{v} the {S} that is not {O}", f"{v} the {S} that isn't {O}",
        f"{v} the {C} {S}, leaving the {O} one untouched", f"leave the {O} {S} alone and {v} the {C} one",
        f"{v} the {C} {S} on the red device", f"{v} the {C} slider on the red switch body",
        f"of the two {S}es, {v} the {C} one" if S.endswith("switch") else f"of the two {S}s, {v} the {C} one",
        f"{v} only the {C} {S}", f"the {C} {S} is the one to {v}", f"{v} the {C} {S}{L} and keep it there",
        f"ignore the {O} {S} and {v} the {C} one", f"{v} the {S} that is {C}, not {O}",
    ]
    return r.choice(frames)


# --------------------------------------------------------------------------- T4
T4_CUBE = ["cube", "cube", "cube", "block", "block", "piece", "brick"]
T4_LEFT = ["left-hand box", "left box", "box on the left", "left container", "left bin", "left-side box",
           "box on the left side", "left tray", "left-hand container"]
T4_RIGHT = ["right-hand box", "right box", "box on the right", "right container", "right bin", "right-side box",
            "box on the right side", "right tray", "right-hand container"]
T4_VERB = ["place", "put", "move", "drop", "set", "transfer", "deposit", "load", "carry"]


def t4(r: random.Random, lab: str) -> str:
    o1, o2 = r.sample([c for c in TASKS["put_blocks_into_boxes"] if c != lab], 2)
    k = r.choice(T4_CUBE)
    ks = k + "s"
    L, R = r.choice(T4_LEFT), r.choice(T4_RIGHT)
    v, v2 = r.choice(T4_VERB), r.choice(T4_VERB)
    X = r.choice([f"the remaining {ks}", f"the other {ks}", "the others", "the rest", f"the other two {ks}",
                  "the other two", "everything else", f"all the other {ks}", "the remaining two",
                  f"the {o1} and {o2} {ks}", f"the {o1} and the {o2} {k}", f"both other {ks}"])
    frames = [
        f"{v} the {lab} {k} in the {L} and {X} in the {R}.", f"{v} the {lab} {k} into the {L} and {X} into the {R}",
        f"{v} {X} in the {R} and the {lab} {k} in the {L}", f"{v} the {o1} and {o2} {ks} in the {R} and the {lab} {k} in the {L}",
        f"{v} the {o1} and {o2} {ks} in the {R}; the {lab} {k} goes in the {L}",
        f"{v} every {k} except the {lab} one in the {R}, and the {lab} one in the {L}",
        f"{v} all {ks} but the {lab} one into the {R}; the {lab} one goes into the {L}",
        f"everything except the {lab} {k} goes into the {R}", f"{lab} {k}: {L}. {o1} and {o2} {ks}: {R}.",
        f"left box: {lab} {k}. right box: {o1} and {o2} {ks}.", f"right box: {o1} and {o2} {ks}. left box: {lab} {k}.",
        f"sort the {ks}: {lab} goes left, {o1} and {o2} go right", f"the {L} gets the {lab} {k}; the {R} gets the others",
        f"the {lab} {k} should go in the {L}, while {X} go in the {R}",
        f"first {v} the {lab} {k} in the {L}, then {v2} {X} in the {R}",
        f"{v} the {k} that is {lab} into the {L} and the others into the {R}",
        f"into the {L} goes the {lab} {k}; into the {R} go the others",
        f"fill the {R} with the {o1} and {o2} {ks} and {v} the {lab} {k} in the {L}",
        f"{v} the {o1} {k} in the {R} along with the {o2} {k}, and the {lab} {k} in the {L}",
        f"{v} the {lab} {k} on the left and {X} on the right", f"{v} the {lab} {k} on the left side and the rest on the right side",
        f"do not put the {lab} {k} in the {R}; it goes in the {L}. the others go in the {R}",
        f"{v} the {o1} {k} in the {L}... no: {v2} the {lab} {k} in the {L} and the rest in the {R}",
        f"{v} the {ks} in the {R}, except the {lab} {k}, which goes in the {L}",
        f"only the {lab} {k} goes in the {L}; {v} {X} in the {R}",
        f"{v} {X} in the {R}, then {v2} the {lab} {k} in the {L}",
        f"the {o1} and {o2} {ks} belong in the {R}, the {lab} {k} belongs in the {L}",
    ]
    return r.choice(frames)


# --------------------------------------------------------------------------- T5
T5_TOK = {"C": ["C", "c"], "o": ["o", "O"], "R": ["R", "r"], "L": ["L", "l"]}
T5_COLOR = {"C": "red", "o": "blue", "R": "yellow", "L": "green"}
T5_NOUN = ["letter", "peg", "piece", "block", "tile", "shape", "letter peg", "letter piece", "letter block"]
T5_VERB = ["insert", "put", "place", "fit", "slot", "push", "drop", "plug", "seat", "set"]
T5_TGT = ["into its slot on the board", "into its slot", "into the board", "into its matching slot",
          "into the matching hole", "in its hole on the board", "into the corresponding slot", "into the right slot",
          "into the slot on the board", "in the board", "into its hole", "into its place on the board",
          "into the matching slot on the board", "into the board where it belongs"]


def _q(r: random.Random, t: str) -> str:
    return r.choice([t, t, t, f"'{t}'", f'"{t}"', f"“{t}”", f"‘{t}’"])


def t5(r: random.Random, lab: str) -> str:
    oth = r.choice([x for x in TASKS["insert_letter"] if x != lab])
    t, to = _q(r, r.choice(T5_TOK[lab])), r.choice(T5_TOK[oth])
    n, v, T = r.choice(T5_NOUN), r.choice(T5_VERB), r.choice(T5_TGT)
    col = T5_COLOR[lab]
    up = r.choice(T5_TOK[lab]).upper() if lab != "o" else r.choice(["O", "o"])
    frames = [
        f"{v} the letter {t} {T}", f"{v} the {t} {n} {T}", f"{v} the {up}-shaped {n} {T}",
        f"{v} the {n} shaped like the letter {t} {T}", f"{v} the {n} in the shape of {a_an(up)} {T}",
        f"{v} the {col} {n} {T}", f"{v} the {col} letter ({t}) {T}", f"{v} the {col} letter {t} {T}",
        f"{v} letter {t} {T}", f"{v} the {t} {T}", f"take the {t} and {v} it {T}",
        f"pick up the letter {t} and {v} it {T}", f"{v} the letter {t}, not the letter {to}, {T}",
        f"{v} the letter {t} (not {to}) {T}", f"of the four letters, {v} the {t} {T}",
        f"leave the {to} alone and {v} the {t} {T}", f"{v} the letter-{t} {n} {T}",
        f"{v} the {'lowercase' if lab == 'o' else 'capital'} {t} {T}", f"{v} the {t} {n}",
        f"the {t} goes {T}: {v} it", f"{v} the letter {t} {T}, ignoring the {to}",
    ]
    return r.choice(frames)


GEN = {"grasp_part": t1, "grasp_move_mug": t2, "toggle_switch_table": t3, "put_blocks_into_boxes": t4,
       "insert_letter": t5}


def _typo(r: random.Random, s: str) -> str:
    words = s.split(" ")
    i = r.randrange(len(words))
    w = words[i]
    if len(w) < 4:
        return s
    j = r.randrange(1, len(w) - 1)
    op = r.choice(["swap", "drop", "dup"])
    if op == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == "drop":
        w = w[:j] + w[j + 1:]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def surface(r: random.Random, s: str) -> str:
    p = r.choice(PREFIX)
    s = s.strip()
    if p:
        s = p + s[0].lower() + s[1:]
    else:
        s = s[0].upper() + s[1:]
    if p in ("Could you ", "Kindly ") and r.random() < 0.6:
        s = s.rstrip(".") + ("?" if p == "Could you " else ".")
    else:
        s = s.rstrip(".") + r.choice(["", "", ".", ".", "!", " ."])
    if r.random() < 0.08:
        s = _typo(r, s)
    u = r.random()
    if u < 0.12:
        s = s.lower()
    elif u < 0.17:
        s = s.upper()
    if r.random() < 0.05:
        s = "  " + s + "  "
    return s


def generate(task: str, n: int, seed: int = 0) -> list[tuple[str, str]]:
    r = random.Random(f"{task}-{seed}")
    out, seen = [], set()
    labels = TASKS[task]
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        lab = labels[len(out) % len(labels)]
        s = surface(r, GEN[task](r, lab))
        if s in seen:
            continue
        seen.add(s)
        out.append((s, lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-task", type=int, default=6000)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for task in TASKS:
            rows = generate(task, args.per_task)
            for s, lab in rows:
                f.write(json.dumps({"task": task, "text": s, "label": lab}) + "\n")
            print(task, len(rows), rows[:3])


if __name__ == "__main__":
    main()
