"""Dev test for mfagent.instructions.parse_variant (CPU only, < 1 s).

  python dev/test_instructions.py

Every official instruction (eval/configs/*.yaml) must map to its variant. The paraphrase
sets come from the 2026-10-07 reviews (core, core held-out, risk, progress evaluation)
plus extra compositional / negation cases. Exits non-zero if an official string fails
or more than MAX_WRONG paraphrases are mis-parsed.
"""
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("METAFINE_REPO") or os.path.join(ROOT, "MetaFine-Challenge_CoRL")  # official repository
sys.path.insert(0, ROOT)
from mfagent.instructions import parse_instruction, parse_variant  # noqa: E402

MAX_WRONG = 0
T1, T2, T3, T4, T5 = "grasp_part", "grasp_move_mug", "toggle_switch_table", "put_blocks_into_boxes", "insert_letter"

CASES = [
    # --- core review set ---
    (T1, "Grasp the cap of the bottle", "cap"), (T1, "Grasp the body of the bottle", "body"),
    (T1, "Pick up the bottle by its cap", "cap"), (T1, "Grab the bottle's lid", "cap"), (T1, "Grasp the bottle cap", "cap"),
    (T1, "grasp the BOTTLE BODY", "body"), (T1, "Hold the bottle by the body, not the cap", "body"),
    (T1, "Grasp the bottle by its middle", "body"), (T1, "Grasp the main body of the bottle, below the cap", "body"),
    (T1, "Grasp the bottle near its top", "cap"), (T1, "Grab the bottle around its side", "body"),
    (T1, "Take hold of the bottle's cap", "cap"), (T1, "Grasp the bottle's neck", "cap"),
    (T1, "Grasp the bottle on the table by the body", "body"), (T1, "Grasp the top part of the bottle", "cap"),
    (T1, "Grasp the cylindrical part of the bottle", "body"), (T1, "Grasp the bottle at its base", "body"),
    (T1, "Grasp the bottle’s body", "body"), (T1, "Pick the bottle up from the top (the cap)", "cap"),
    (T1, "Do not grasp the cap; grasp the body of the bottle", "body"),
    (T2, "Grasp the mug by the handle and move it to the left", "left"),
    (T2, "Grasp the mug by the handle and move it to the right", "right"),
    (T2, "Grasp the mug by the handle and move it forward", "forward"),
    (T2, "Pick up the mug by its handle and shift it left", "left"),
    (T2, "Grab the mug handle and slide the mug rightwards", "right"), (T2, "Grasp the mug handle and move it leftward", "left"),
    (T2, "Move the mug to the right after grasping its handle", "right"),
    (T2, "Grasp the mug by the handle and push it forwards", "forward"),
    (T2, "Grasp the mug by the handle and move it away from you", "forward"),
    (T2, "Grasp the mug by the handle and move it to the front", "forward"),
    (T2, "Grasp the mug's handle; move it 10 cm to the left side", "left"),
    (T2, "Grasp the cup by its handle and move it right", "right"),
    (T2, "Holding the handle, move the mug forward, not left", "forward"),
    (T2, "Grasp the mug by the handle (on its right side) and move it to the left", "left"),
    (T2, "Grasp the mug by the handle and move it toward the left", "left"),
    (T2, "Grasp the mug by the handle and move it in the forward direction", "forward"),
    (T2, "Grasp the mug by the handle and move it to the LEFT", "left"),
    (T2, "Grasp the mug by the handle and move it ahead", "forward"),
    (T3, "Toggle the red switch on the table", "red"), (T3, "Toggle the blue switch on the table", "blue"),
    (T3, "Flip the blue switch", "blue"), (T3, "Push the red slider", "red"), (T3, "Toggle the switch that is blue", "blue"),
    (T3, "Toggle the blue switch, not the red one", "blue"),
    (T3, "Toggle the switch with the blue button on the red base", "blue"), (T3, "Turn on the BLUE switch", "blue"),
    (T3, "Switch the red toggle on the table", "red"), (T3, "Toggle the reddish switch", "red"),
    (T3, "Toggle the blue-coloured switch", "blue"), (T3, "Toggle the switch colored blue", "blue"),
    (T3, "Slide the red switch on the red device", "red"), (T3, "Toggle the light-blue switch", "blue"),
    (T3, "On the red switch box, toggle the blue switch", "blue"),
    (T4, "Place the red cube in the left-hand box and the remaining cubes in the right-hand box.", "red"),
    (T4, "Place the blue cube in the left-hand box and the remaining cubes in the right-hand box.", "blue"),
    (T4, "Place the green cube in the left-hand box and the remaining cubes in the right-hand box.", "green"),
    (T4, "Put the blue block into the left box and the other blocks into the right box", "blue"),
    (T4, "Put the red and green cubes in the right box and the blue cube in the left box", "blue"),
    (T4, "Put the green and blue cubes in the right box, and the red cube in the left box", "red"),
    (T4, "Place the remaining cubes in the right-hand box and the red cube in the left-hand box.", "red"),
    (T4, "Move the RED cube to the left box; the rest go to the right box.", "red"),
    (T4, "Place all cubes except the blue one in the right-hand box, and the blue one in the left-hand box", "blue"),
    (T4, "Place the blue cube on the left, the red and green cubes on the right", "blue"),
    (T4, "Sort the cubes: green goes left, red and blue go right", "green"),
    (T4, "The left-hand box gets the blue cube; the right-hand box gets the others", "blue"),
    (T4, "Put every cube except the red one into the right box; put the red one in the left box", "red"),
    (T4, "Place the cube that is green into the left box and the others into the right box", "green"),
    (T4, "Put the blue cube in the box on the left and the red and green cubes in the box on the right", "blue"),
    (T4, "Place the red cube in the right-hand box... no: place the green cube in the left-hand box and the rest in the right-hand box", "green"),
    (T5, "Insert the letter C into its slot on the board", "C"), (T5, "Insert the letter o into its slot on the board", "o"),
    (T5, "Insert the letter R into its slot on the board", "R"), (T5, "Insert the letter L into its slot on the board", "L"),
    (T5, "Insert the letter O into its slot on the board", "o"), (T5, "Insert the letter 'R' into its slot", "R"),
    (T5, "Insert the letter “L” into the board", "L"), (T5, "Insert the O-shaped peg into its slot", "o"),
    (T5, "Put the L peg into its slot on the board", "L"), (T5, "Insert letter r into the board", "R"),
    (T5, "Insert the 'o' letter peg into its hole", "o"), (T5, "Place the letter-C peg in the matching slot", "C"),
    (T5, "Insert the peg shaped like the letter R", "R"), (T5, "Insert the yellow letter R into its slot", "R"),
    (T5, "Insert the green letter (L) into the board", "L"), (T5, "Insert the C into its slot", "C"),
    (T5, "Insert the R peg", "R"), (T5, "Insert the letter L, not the letter C, into its slot", "L"),
    (T5, "insert the letter l into its slot on the board", "L"), (T5, "Insert the lowercase letter o into its slot", "o"),
    # --- core held-out set ---
    (T1, "Pick up the bottle by grabbing its cap", "cap"), (T1, "Grab the bottle by the lid", "cap"),
    (T1, "Grasp the bottle, holding it by the body", "body"), (T1, "Grip the body part of the bottle", "body"),
    (T1, "Grasp the bottle's cap, not its body", "cap"), (T1, "Please grasp the bottle body.", "body"),
    (T2, "Pick the mug up by its handle and move it towards the left", "left"),
    (T2, "Grasp the handle of the mug, then move the mug to your right", "right"),
    (T2, "Take the mug by the handle and move it forward by 10 cm", "forward"),
    (T2, "Grasp the mug by the handle and move it to the left by ten centimetres", "left"),
    (T2, "Grasp the mug handle and translate the mug rightward", "right"),
    (T2, "Grasp the mug by the handle, move it forward", "forward"), (T2, "Move the mug forward while holding its handle", "forward"),
    (T3, "Toggle the blue switch", "blue"), (T3, "Press the red switch on the table", "red"),
    (T3, "Flip the switch whose slider is blue", "blue"), (T3, "Toggle the red one", "red"),
    (T3, "Toggle the blue switch on the red device", "blue"),
    (T4, "Put the red cube into the left box and the others into the right box", "red"),
    (T4, "Place the green block in the left container and the other blocks in the right container", "green"),
    (T4, "Place the blue cube in the left-hand box; place the red and green cubes in the right-hand box", "blue"),
    (T4, "Put the cubes in the right-hand box, except the green cube, which goes in the left-hand box", "green"),
    (T4, "Red cube: left box. Blue and green cubes: right box.", "red"),
    (T4, "Place the red and blue cubes in the right box, then place the green cube in the left box", "green"),
    (T5, "Insert the letter R into its slot", "R"), (T5, "Put the letter o in the board", "o"),
    (T5, "Insert the 'C' peg into the board", "C"), (T5, "Insert letter L in its matching hole", "L"),
    (T5, "Insert the R-shaped piece into its slot on the board", "R"), (T5, "Slot the letter O into the board", "o"),
    (T5, "Insert the blue letter into its slot", "o"), (T5, "Insert the yellow peg into its slot", "R"),
    (T5, "Pick up the letter C and insert it into the board", "C"),
    # --- risk review set ---
    (T1, "Grab the bottle cap", "cap"), (T1, "Hold the bottle around its body", "body"),
    (T1, "Grasp the bottle by the middle", "body"), (T1, "Grasp the bottle's main body, not the cap", "body"),
    (T1, "Grasp the lid of the bottle", "cap"), (T1, "Grip the bottle body", "body"), (T1, "Grasp the bottle below the cap", "body"),
    (T1, "Grasp the bottle's base", "body"), (T1, "Grasp the bottle by its lower part", "body"), (T1, "Grasp the bottle's top", "cap"),
    (T1, "Grasp the bottle around the label", "body"), (T1, "Grasp the bottle around its torso", "body"), (T1, "Grasp the cap", "cap"),
    (T2, "Pick up the mug by its handle and shift it leftward", "left"), (T2, "Grab the mug handle and slide it leftwards", "left"),
    (T2, "Grasp the mug by the handle and move it rightwards", "right"),
    (T2, "Hold the mug by the handle and move it toward the right side", "right"),
    (T2, "Grasp the mug by the handle and move it forwards", "forward"),
    (T2, "Grasp the mug by the handle and push it away from the robot", "forward"),
    (T2, "Grasp the mug handle and move it toward the camera", "forward"),
    (T2, "Grasp the mug by the handle and move it 10 cm forward", "forward"),
    (T2, "Grasp the mug by the handle and move it further from you", "forward"),
    (T2, "Grasp the mug by the handle (not the left side) and move it to the right", "right"),
    (T3, "Flip the switch that is blue", "blue"), (T3, "Slide the blue slider", "blue"), (T3, "Push the red toggle", "red"),
    (T3, "Toggle the bluish switch", "blue"), (T3, "Do not touch the red switch; toggle the blue one", "blue"),
    (T4, "Put the red block into the left box and the other blocks into the right box", "red"),
    (T4, "Put the green and blue cubes in the right box and the red cube in the left box", "red"),
    (T4, "Place every cube except the blue one in the right-hand box; the blue one goes left", "blue"),
    (T4, "Move the green cube to the left container and the rest to the right container", "green"),
    (T4, "Place the blue block on the left and the red and green blocks on the right", "blue"),
    (T5, "Insert the R into its slot on the board", "R"), (T5, "Put the L peg into the matching hole", "L"),
    (T5, "Insert the lowercase o into the board", "o"), (T5, "Insert the 'R' letter into its slot", "R"),
    (T5, "Place letter-L into its slot", "L"), (T5, 'Insert the letter "o" into its slot on the board', "o"),
    (T5, "Insert the green letter into its slot", "L"), (T5, "Insert the red letter into its slot", "C"),
    # --- progress-evaluation set ---
    (T1, "Pick up the bottle by its body", "body"), (T1, "Hold the bottle cap", "cap"),
    (T1, "Take hold of the bottle by the body, not the cap", "body"),
    (T2, "Pick up the mug by its handle and slide it to the right", "right"),
    (T2, "Hold the handle of the mug and move the mug leftward", "left"),
    (T3, "Flip the blue-coloured switch", "blue"),
    (T4, "The blue block goes to the left box; the others go to the right box", "blue"),
    (T5, "Insert the letter 'L' into its slot", "L"), (T5, "Insert the letter “o” into its slot on the board", "o"),
    (T5, "Put the L-shaped peg into the matching slot", "L"),
    # --- extra: compositional commands, negation, formatting ---
    (T1, "On top of the table, grasp the bottle's body", "body"), (T1, "Grasp the bottle cap rather than the body", "cap"),
    (T1, "Avoid the cap and grasp the bottle around its middle", "body"), (T1, "  GRASP THE CAP OF THE BOTTLE  ", "cap"),
    (T1, "Grasp the body of the bottle.", "body"), (T1, "Don't grasp the body, grasp the cap", "cap"),
    (T2, "Grasp the mug by the handle and move it to the right.", "right"), (T2, "Grab the mug's handle and move it ahead of you", "forward"),
    (T2, "Grasp the handle of the mug on the left and move the mug forward", "forward"),
    (T2, "Grasp the mug by its handle and carry it rightward by about ten centimetres", "right"),
    (T3, "Toggle the switch that is not red", "blue"), (T3, "Toggle the blue slider on the red switch body", "blue"),
    (T3, "TOGGLE THE RED SWITCH ON THE TABLE", "red"),
    (T4, "First put the red cube in the left box, then put the green and blue cubes in the right box", "red"),
    (T4, "Put the green and red cubes in the right-hand box; the blue cube belongs in the left-hand box.", "blue"),
    (T4, "Left box: green cube. Right box: red and blue cubes.", "green"),
    (T4, "Everything except the red cube goes into the right-hand box", "red"),
    (T4, "place the blue cube in the left-hand box and the remaining cubes in the right-hand box", "blue"),
    (T5, "Insert the letter ‘R’ into its slot on the board", "R"), (T5, "Insert the letter-o peg into the board", "o"),
    (T5, "Take the L and put it in its slot", "L"), (T5, "Insert the letter C, not the R, into the board", "C"),
    (T5, "INSERT THE LETTER O INTO ITS SLOT ON THE BOARD", "o"),
    # --- second held-out set (written before testing; 2 general fixes followed: "isn't", body descriptors) ---
    (T1, "Lift the bottle by holding its cap", "cap"), (T1, "Use the gripper to seize the bottle's body", "body"),
    (T1, "Grasp the bottle near the bottom", "body"), (T1, "Close the gripper around the bottle's cap", "cap"),
    (T1, "Grasp the bottle at its widest part", "body"), (T1, "Grab the bottle by the closure", "cap"),
    (T2, "Grip the mug handle and drag the mug to the left-hand side", "left"),
    (T2, "Holding the mug by the handle, bring it forward a little", "forward"),
    (T2, "Grasp the handle and move the mug toward your right", "right"),
    (T2, "Seize the mug by its handle, then shift the mug sideways to the right", "right"),
    (T2, "Grab the mug by the handle and move it away from the base of the robot", "forward"),
    (T2, "Grasp the mug by the handle and slide it in the leftward direction", "left"),
    (T2, "Grasp the mug by the handle and move it to the left, not to the right", "left"),
    (T3, "Actuate the blue switch", "blue"), (T3, "Toggle the switch with the red slider", "red"),
    (T3, "Please flip the blue toggle switch on the table", "blue"), (T3, "Toggle the switch that isn't red", "blue"),
    (T3, "Toggle the red switch, leaving the blue one untouched", "red"),
    (T4, "Drop the green cube into the left box, and drop the other two into the right box", "green"),
    (T4, "The red cube should go in the left-hand box, while the blue and green cubes go in the right-hand box", "red"),
    (T4, "Fill the right box with the red and blue cubes and put the green cube in the left box", "green"),
    (T4, "Put the blue cube on the left side and the rest on the right side", "blue"),
    (T4, "Into the left-hand box goes the red cube; into the right-hand box go the others", "red"),
    (T4, "Put the red cube in the right box along with the blue cube, and the green cube in the left box", "green"),
    (T5, "Fit the letter R into the board", "R"), (T5, "Insert the capital L into its slot", "L"),
    (T5, "Push the O into its hole on the board", "o"), (T5, "Insert the peg in the shape of a C", "C"),
    (T5, "Insert the yellow piece into its slot", "R"), (T5, "Place the R block into its matching slot on the board", "R"),
]


def main():
    wrong, n_off = [], 0
    for path in sorted(glob.glob(os.path.join(REPO, "eval", "configs", "*.yaml"))):
        cfg = yaml.safe_load(open(path))
        for variant, text in (cfg.get("variants") or {}).items():
            n_off += 1
            got, how = parse_instruction(cfg["task_id"], text)
            if got != variant or how != "exact":
                print(f"OFFICIAL WRONG {cfg['task_id']}: {text!r} -> {got} ({how}; want {variant})")
                return 1
    seen = set()
    for task, text, want in CASES:
        if (task, text) in seen:
            continue
        seen.add((task, text))
        got, how = parse_instruction(task, text)
        if got != want or how == "default":  # a correct answer must not be the default by luck
            wrong.append((task, text, want, f"{got} ({how})"))
    for task, text, want, got in wrong:
        print(f"WRONG {task:22s} want={want:8s} got={got:8s} | {text}")
    for task in ("grasp_part", "grasp_move_mug", "toggle_switch_table", "put_blocks_into_boxes", "insert_letter", "nope"):
        for junk in (None, "", "   ", "x" * 100000, "🤖🫙🔴", 12345, b"bytes"):  # must never raise
            parse_variant(task, junk)
            if parse_instruction(task, junk)[1] != "default":
                print(f"JUNK PARSED {task}: {str(junk)[:40]!r} -> {parse_instruction(task, junk)}")
                return 1
    print(f"official strings: {n_off}/{n_off} OK | paraphrases: {len(seen) - len(wrong)}/{len(seen)} OK | junk inputs OK")
    return 0 if len(wrong) <= MAX_WRONG else 1


if __name__ == "__main__":
    sys.exit(main())
