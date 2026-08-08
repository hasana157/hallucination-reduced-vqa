"""Evaluation utilities: answer normalization, text extraction, MSCOCO vocabulary, and reproducibility checking.

Implements standard normalization for VQA accuracy calculation, yes/no extraction
for POPE evaluation, object mention extraction for CHAIR evaluation, and
the ReproducibilityChecker per SRS NFG-4.
"""

import logging
import re
import string
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# MSCOCO 80 Object Classes & Synonyms for CHAIR Metric
# ============================================================
MSCOCO_80_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Map plural / alternate forms to canonical COCO class names
MSCOCO_SYNONYMS: Dict[str, str] = {
    "people": "person", "man": "person", "woman": "person", "boy": "person",
    "girl": "person", "child": "person", "kid": "person", "persons": "person",
    "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "cars": "car", "automobile": "car", "automobiles": "car",
    "motorcycles": "motorcycle", "motorbike": "motorcycle", "motorbikes": "motorcycle",
    "airplanes": "airplane", "plane": "airplane", "planes": "airplane",
    "buses": "bus", "busses": "bus",
    "trains": "train",
    "trucks": "truck",
    "boats": "boat", "ship": "boat", "ships": "boat",
    "traffic lights": "traffic light",
    "fire hydrants": "fire hydrant",
    "stop signs": "stop sign",
    "parking meters": "parking meter",
    "benches": "bench",
    "birds": "bird",
    "cats": "cat", "kitten": "cat", "kittens": "cat",
    "dogs": "dog", "puppy": "dog", "puppies": "dog",
    "horses": "horse",
    "sheeps": "sheep",
    "cows": "cow",
    "elephants": "elephant",
    "bears": "bear",
    "zebras": "zebra",
    "giraffes": "giraffe",
    "backpacks": "backpack", "bag": "backpack", "bags": "backpack",
    "umbrellas": "umbrella",
    "handbags": "handbag", "purse": "handbag", "purses": "handbag",
    "ties": "tie",
    "suitcases": "suitcase", "luggage": "suitcase",
    "frisbees": "frisbee",
    "skis": "skis", "ski": "skis",
    "snowboards": "snowboard",
    "balls": "sports ball", "ball": "sports ball", "sports balls": "sports ball",
    "kites": "kite",
    "baseball bats": "baseball bat",
    "baseball gloves": "baseball glove",
    "skateboards": "skateboard",
    "surfboards": "surfboard",
    "tennis rackets": "tennis racket", "racket": "tennis racket", "rackets": "tennis racket",
    "bottles": "bottle",
    "wine glasses": "wine glass",
    "cups": "cup", "mug": "cup", "mugs": "cup",
    "forks": "fork",
    "knives": "knife",
    "spoons": "spoon",
    "bowls": "bowl",
    "bananas": "banana",
    "apples": "apple",
    "sandwiches": "sandwich",
    "oranges": "orange",
    "broccolis": "broccoli",
    "carrots": "carrot",
    "hot dogs": "hot dog",
    "pizzas": "pizza",
    "donuts": "donut", "doughnut": "donut", "doughnuts": "donut",
    "cakes": "cake",
    "chairs": "chair", "seat": "chair", "seats": "chair",
    "couches": "couch", "sofa": "couch", "sofas": "couch",
    "potted plants": "potted plant", "plant": "potted plant", "plants": "potted plant",
    "beds": "bed",
    "tables": "dining table", "dining tables": "dining table", "table": "dining table",
    "toilets": "toilet",
    "tvs": "tv", "television": "tv", "televisions": "tv", "monitor": "tv", "monitors": "tv",
    "laptops": "laptop", "computer": "laptop", "computers": "laptop",
    "mice": "mouse",
    "remotes": "remote",
    "keyboards": "keyboard",
    "cell phones": "cell phone", "phone": "cell phone", "phones": "cell phone", "mobile phone": "cell phone",
    "microwaves": "microwave",
    "ovens": "oven",
    "toasters": "toaster",
    "sinks": "sink",
    "refrigerators": "refrigerator", "fridge": "refrigerator", "fridges": "refrigerator",
    "books": "book",
    "clocks": "clock",
    "vases": "vase",
    "scissors": "scissors",
    "teddy bears": "teddy bear",
    "hair driers": "hair drier", "hairdryer": "hair drier", "hairdryers": "hair drier",
    "toothbrushes": "toothbrush",
}

# Number word to digit mapping
WORD_TO_DIGIT: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

# Contraction mapping
CONTRACTIONS: Dict[str, str] = {
    "aint": "is not", "cannot": "can not", "cant": "can not", "couldnt": "could not",
    "dont": "do not", "doesnt": "does not", "hadnt": "had not", "hasnt": "has not",
    "havent": "have not", "isnt": "is not", "didnt": "did not", "shouldnt": "should not",
    "wasnt": "was not", "werent": "were not", "wont": "will not", "wouldnt": "would not",
}

# Articles to strip
ARTICLES: Set[str] = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """Official VQAv2 answer normalization protocol.

    Steps:
      1. Lowercase text.
      2. Strip punctuation.
      3. Normalize contractions.
      4. Remove articles ('a', 'an', 'the').
      5. Convert number words to digits ('two' -> '2').
      6. Collapse whitespace.

    Args:
        answer: Raw generated or target answer string.

    Returns:
        Normalized answer string.
    """
    if not answer or not isinstance(answer, str):
        return ""

    text = answer.lower().strip()

    # Normalize contractions
    words = []
    for word in text.split():
        word_clean = word.translate(str.maketrans("", "", string.punctuation))
        if word_clean in CONTRACTIONS:
            words.extend(CONTRACTIONS[word_clean].split())
        else:
            words.append(word)
    text = " ".join(words)

    # Remove punctuation except between digits
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Strip articles and normalize numbers
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        if token in ARTICLES:
            continue
        token = WORD_TO_DIGIT.get(token, token)
        normalized_tokens.append(token)

    return " ".join(normalized_tokens).strip()


def extract_yes_no(text: str) -> Optional[str]:
    """Robustly extract 'yes' or 'no' from a generated text answer for POPE evaluation.

    Args:
        text: Raw answer string from model output.

    Returns:
        "yes", "no", or None if neither can be confidently identified.
    """
    if not text:
        return None

    clean = normalize_answer(text)

    # Direct exact match
    if clean in ("yes", "no"):
        return clean

    # First token check
    tokens = clean.split()
    if tokens:
        if tokens[0] in ("yes", "yeah", "yep", "true", "correct"):
            return "yes"
        if tokens[0] in ("no", "nope", "false", "incorrect"):
            return "no"

    # Regex substring matching for phrases like "there is a cat" vs "there is no cat"
    if re.search(r"\b(yes|there is|there are|can see|i see)\b", clean):
        if not re.search(r"\b(no|not|cannot|cant|don\'t|doesn\'t|none)\b", clean):
            return "yes"

    if re.search(r"\b(no|there is no|there are no|cannot see|not present|isn\'t|aren\'t)\b", clean):
        return "no"

    return None


def extract_object_mentions(
    text: str,
    coco_classes: Optional[List[str]] = None,
    synonyms: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Extract mentioned MSCOCO object names from generated text for CHAIR scoring.

    Args:
        text: Generated answer text.
        coco_classes: Optional list of canonical object names. Defaults to MSCOCO_80_CLASSES.
        synonyms: Optional map of synonym->canonical name. Defaults to MSCOCO_SYNONYMS.

    Returns:
        List of canonical COCO object class strings detected in the text.
    """
    if not text:
        return []

    classes = coco_classes or MSCOCO_80_CLASSES
    syn_map = synonyms or MSCOCO_SYNONYMS

    clean = normalize_answer(text)
    tokens = clean.split()

    detected: Set[str] = set()

    # Check multi-word object classes first (e.g., "traffic light", "tennis racket")
    for obj in classes:
        if " " in obj and obj in clean:
            detected.add(obj)

    # Check single-token words and synonyms
    for token in tokens:
        if token in classes:
            detected.add(token)
        elif token in syn_map:
            detected.add(syn_map[token])

    # Check bigrams for multi-word synonyms
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in syn_map:
            detected.add(syn_map[bigram])

    return list(detected)


class ReproducibilityChecker:
    """Runs evaluation multiple times with seed=42 to verify reproducibility <= 0.1% variance.

    Per SRS NFG-4: "Evaluation results must be reproducible within +/-0.1% variance across repeated runs."
    """

    def __init__(self, tolerance_pct: float = 0.1, n_runs: int = 3) -> None:
        self.tolerance_pct = tolerance_pct
        self.n_runs = n_runs

    def check(
        self,
        eval_fn: Callable[[], Dict[str, float]],
    ) -> Dict[str, Any]:
        """Run eval_fn multiple times and calculate metric variances.

        Args:
            eval_fn: Function that runs evaluation and returns a metric name -> float dict.

        Returns:
            Dict containing:
                - passed (bool): True if max_variance_pct <= tolerance_pct for all metrics.
                - metrics (dict): Detailed stats (mean, std, min, max, variance_pct) per metric.
                - tolerance_pct (float): Target threshold.
        """
        all_results: List[Dict[str, float]] = []

        logger.info(f"Running ReproducibilityChecker across {self.n_runs} runs...")
        for run_idx in range(self.n_runs):
            res = eval_fn()
            all_results.append(res)
            logger.info(f"Reproducibility run {run_idx + 1}/{self.n_runs}: {res}")

        if not all_results:
            return {"passed": False, "metrics": {}, "error": "No results collected"}

        metric_keys = all_results[0].keys()
        stats: Dict[str, Dict[str, float]] = {}
        all_passed = True

        for k in metric_keys:
            vals = [r[k] for r in all_results if k in r and r[k] is not None]
            if not vals:
                continue

            arr = np.array(vals, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))

            # Variance percentage relative to mean (range / mean * 100 or std / mean * 100)
            range_val = max_val - min_val
            variance_pct = (range_val / mean_val * 100.0) if mean_val != 0 else 0.0

            passed_metric = variance_pct <= self.tolerance_pct
            if not passed_metric:
                all_passed = False

            stats[k] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 6),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "variance_pct": round(variance_pct, 4),
                "passed": passed_metric,
            }

        logger.info(f"Reproducibility check completed: passed={all_passed}")

        return {
            "passed": all_passed,
            "metrics": stats,
            "tolerance_pct": self.tolerance_pct,
            "n_runs": self.n_runs,
        }
