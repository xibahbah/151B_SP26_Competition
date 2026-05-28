"""Postprocessing and error analysis helpers for FRQ evaluation."""

from __future__ import annotations

import re
from typing import Any


UNIT_WORDS = (
    "degrees fahrenheit",
    "degrees celsius",
    "degrees kelvin",
    "degrees rankine",
    "fahrenheit",
    "celsius",
    "kelvin",
    "rankine",
    "degrees",
    "degree",
    "hours",
    "hour",
    "minutes",
    "minute",
    "seconds",
    "second",
    "years",
    "year",
    "feet",
    "foot",
    "meters",
    "meter",
    "miles",
    "mile",
    "dollars",
    "dollar",
)


def _find_boxed_entries(text: str) -> list[tuple[int, int, str]]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            entries.append((idx, i, text[brace_start : i - 1].strip()))
        start = max(i, idx + 1)
    return entries


def extract_final_answer_text(response: str) -> tuple[str, list[str]]:
    """Extract the most likely final answer text without changing its math."""
    notes: list[str] = []
    text = response.strip()
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :].strip()
        notes.append("removed_think_prefix")

    entries = _find_boxed_entries(text)
    if entries:
        last_group = [entries[-1]]
        for i in range(len(entries) - 2, -1, -1):
            gap = text[entries[i][1] : entries[i + 1][0]]
            if re.fullmatch(r"[\s,\$.;:\-&\\]*", gap or ""):
                last_group.insert(0, entries[i])
            else:
                break
        notes.append("extracted_boxed")
        return ", ".join(entry[2] for entry in last_group), notes

    marker_patterns = [
        r"FINAL\s*:",
        r"Final answer\s*:",
        r"final answer\s*:",
        r"Answer\s*:",
        r"answer\s*:",
        r"answer is",
        r"####",
        r"# Answer",
    ]
    for pattern in marker_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            notes.append("extracted_marker")
            return text[matches[-1].end() :].strip(), notes

    notes.append("no_explicit_answer")
    return text, notes


def _question_requires_dollar(question: str) -> bool:
    q = question.lower()
    return "must begin with a dollar sign" in q or "must begin with $" in q


def _question_requires_percent(question: str) -> bool:
    q = question.lower()
    return "fill in the blank with a percent" in q or "include %" in q


def _remove_thousands_commas(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)


def _strip_outer_box_or_dollars(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        text = text[1:-1].strip()
    return text


def _normalize_separators(text: str, expected_count: int) -> str:
    text = text.replace("|||", ",")
    if expected_count > 1:
        text = re.sub(r"[\n;]+", ",", text)
    else:
        text = text.replace("\n", " ")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" ,.")


def _split_top_level_commas(text: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closers = set(pairs.values())
    for i, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closers and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _strip_units_from_part(part: str, question: str) -> str:
    if _question_requires_dollar(question) or _question_requires_percent(question):
        preserve_symbols = True
    else:
        preserve_symbols = False

    out = part.strip()
    if not _question_requires_dollar(question):
        out = re.sub(r"^\$\s*", "", out)
    if not _question_requires_percent(question):
        out = re.sub(r"\s*percent$", "", out, flags=re.IGNORECASE)
    if not preserve_symbols:
        out = re.sub(r"\s+(?:%s)\.?$" % "|".join(re.escape(u) for u in UNIT_WORDS), "", out, flags=re.IGNORECASE)
    return out.strip()


def postprocess_response(response: str, question: str, expected_count: int) -> dict[str, Any]:
    answer_text, notes = extract_final_answer_text(response)
    answer_text = _strip_outer_box_or_dollars(answer_text)
    answer_text = _remove_thousands_commas(answer_text)
    answer_text = _normalize_separators(answer_text, expected_count)
    parts = _split_top_level_commas(answer_text) if expected_count > 1 else [answer_text.strip()]
    parts = [_strip_units_from_part(part, question) for part in parts]
    answer_text = ", ".join(part for part in parts if part != "")
    boxed = f"\\boxed{{{answer_text}}}"
    return {
        "answer_text": answer_text,
        "response": boxed,
        "notes": notes,
    }


def _gold_list(gold: Any) -> list[str]:
    if isinstance(gold, list):
        return [str(x) for x in gold]
    return [str(gold)]


def _safe_judge(judger: Any, pred: str, gold: Any) -> bool:
    gold_items = _gold_list(gold)
    try:
        return bool(judger.auto_judge(pred=pred, gold=gold_items, options=[[]] * len(gold_items)))
    except Exception:
        return False


def _answer_count(judger: Any, pred: str) -> int:
    try:
        extracted = judger.extract_ans(pred)
        return len(judger.split_by_comma(extracted)) if extracted else 0
    except Exception:
        return 0


def _looks_like_precision_issue(answer_text: str, gold: Any) -> bool:
    gold_items = _gold_list(gold)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    if len(pred_nums) != len(gold_items):
        return False
    for pred, ref in zip(pred_nums, gold_items):
        try:
            p = float(pred)
            g = float(ref)
        except ValueError:
            return False
        if abs(p - g) > max(1e-2, abs(g) * 1e-4):
            return False
    return True


def classify_error(judger: Any, raw_response: str, postprocessed: dict[str, Any], gold: Any, raw_correct: bool, final_correct: bool) -> str:
    if final_correct:
        return "correct_after_postprocess" if not raw_correct else "correct"
    if "no_explicit_answer" in postprocessed["notes"]:
        return "missing_boxed"
    expected_count = len(_gold_list(gold))
    if _answer_count(judger, postprocessed["response"]) != expected_count:
        return "wrong_answer_count"
    if _looks_like_precision_issue(postprocessed["answer_text"], gold):
        return "precision_rounding"
    if any(re.search(r"[A-Za-z_^*/()+-]", item) for item in _gold_list(gold)):
        return "expression_format"
    return "wrong_math"


def score_frq_item(judger: Any, item: dict[str, Any], response: str) -> dict[str, Any]:
    gold = item["answer"]
    gold_items = _gold_list(gold)
    raw_correct = _safe_judge(judger, response, gold_items)
    postprocessed = postprocess_response(response, item["question"], len(gold_items))
    final_correct = _safe_judge(judger, postprocessed["response"], gold_items)
    return {
        "postprocessed_response": postprocessed["response"],
        "postprocessed_answer": postprocessed["answer_text"],
        "postprocess_notes": postprocessed["notes"],
        "raw_correct": raw_correct,
        "correct": final_correct,
        "error_type": classify_error(judger, response, postprocessed, gold_items, raw_correct, final_correct),
    }
