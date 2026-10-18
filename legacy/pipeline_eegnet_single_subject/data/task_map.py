"""
eegmmidb run/annotation reference.

Each subject has 14 runs. Runs 1-2 are baseline (no task) and are skipped.
The remaining 12 alternate between two task families, each repeated 3x:

  Runs 3, 7, 11  -> Motor Execution:  left fist (T1) vs right fist (T2)
  Runs 4, 8, 12  -> Motor Imagery:    left fist (T1) vs right fist (T2)
  Runs 5, 9, 13  -> Motor Execution:  both fists (T1) vs both feet (T2)
  Runs 6, 10, 14 -> Motor Imagery:    both fists (T1) vs both feet (T2)

T0 is rest in every run.
"""

LEFT_RIGHT_FIST_RUNS = {"ME": [3, 7, 11], "MI": [4, 8, 12]}
FISTS_FEET_RUNS = {"ME": [5, 9, 13], "MI": [6, 10, 14]}

LEFT_RIGHT_LABELS = {"T0": "rest", "T1": "left", "T2": "right"}
FISTS_FEET_LABELS = {"T0": "rest", "T1": "both_fists", "T2": "both_feet"}

# Subjects 88, 92, 100, 104 have a well-documented equipment/annotation-timing
# fault in eegmmidb (different sampling rate / corrupted event markers) and
# are routinely excluded in BCI benchmarks using this dataset. 106 is
# sometimes flagged too. Excluded by default in load_subjects().
KNOWN_BAD_SUBJECTS = [88, 92, 100, 104, 106]


def run_family(run_number: int) -> str:
    if run_number in (1, 2):
        return "baseline"
    if run_number in LEFT_RIGHT_FIST_RUNS["ME"] + LEFT_RIGHT_FIST_RUNS["MI"]:
        return "left_right"
    if run_number in FISTS_FEET_RUNS["ME"] + FISTS_FEET_RUNS["MI"]:
        return "fists_feet"
    raise ValueError(f"Unrecognized run number: {run_number}")


def run_mode(run_number: int) -> str:
    if run_number in LEFT_RIGHT_FIST_RUNS["ME"] + FISTS_FEET_RUNS["ME"]:
        return "ME"
    if run_number in LEFT_RIGHT_FIST_RUNS["MI"] + FISTS_FEET_RUNS["MI"]:
        return "MI"
    raise ValueError(f"Run {run_number} has no ME/MI mode (baseline run)")


def labels_for_run(run_number: int) -> dict:
    family = run_family(run_number)
    if family == "left_right":
        return LEFT_RIGHT_LABELS
    if family == "fists_feet":
        return FISTS_FEET_LABELS
    raise ValueError(f"Run {run_number} is a baseline run — no T1/T2 labels")
