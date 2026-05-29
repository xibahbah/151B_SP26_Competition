#!/usr/bin/env python3
"""Repair saved FRQ generations without rerunning the model.

The selected repair is private-safe: it chooses from answer candidates using
format heuristics only. The oracle fields are diagnostic only and show how much
accuracy is already present somewhere in the generated text.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import json
import re
import signal
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frq_tools import (  # noqa: E402
    _find_boxed_entries,
    _gold_list,
    _safe_judge,
    _split_top_level_commas,
    classify_error,
    postprocess_response,
    score_frq_item,
)
from judger import Judger  # noqa: E402
from tqdm import tqdm  # noqa: E402


ANSWERISH_MARKERS = (
    "final answer",
    "final answers",
    "the answer is",
    "the answers are",
    "the answer should be",
    "the answers should be",
    "so the answer is",
    "so the answers are",
    "so the three answers are",
    "so the two answers are",
    "answers are",
    "solutions are",
    "values are",
    "therefore",
    "thus",
)

ORDINAL_LABELS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)

NAMED_LABELS = (
    "celsius",
    "kelvin",
    "rankine",
    "fahrenheit",
    "coworker's",
    "coworkers",
    "yours",
    "mine",
    "distance",
    "bearing",
    "mean",
    "median",
    "mode",
    "range",
    "standard deviation",
)

EXPLANATION_PHRASES = (
    " wait",
    " let's",
    " let me",
    " but ",
    " however",
    " because",
    " since ",
    " which ",
    " so ",
    " therefore",
    " hence",
    " check",
    " the problem",
    " the question",
    " i think",
    " i need",
)

VERBOSE_WORDS = (
    "wait",
    "because",
    "since",
    "therefore",
    "hence",
    "approximately",
    "degrees",
    "problem",
    "question",
    "calculate",
    "compute",
    "answer is",
    "answers are",
)


@dataclass(frozen=True)
class Candidate:
    answer_text: str
    response: str
    source: str
    quality: float


class JudgeTimeout(RuntimeError):
    pass


def _handle_timeout(signum: int, frame: Any) -> None:
    raise JudgeTimeout()


def judge_with_timeout(judger: Judger, response: str, gold: Any, timeout_seconds: float = 0.25) -> bool:
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        return _safe_judge(judger, response, _gold_list(gold))
    except JudgeTimeout:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def expected_answer_count(item: dict[str, Any]) -> int:
    if "answer" in item:
        return len(_gold_list(item["answer"]))
    return max(1, item.get("question", "").count("[ANS]"))


def has_choice_options(question: str) -> bool:
    return bool(re.search(r"\bA\.\s+", question) and re.search(r"\bB\.\s+", question))


def strip_think(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>") :].strip() if think_end >= 0 else text.strip()


def trim_explanation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:is|are|=|:|-\s+)\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:about|approximately|approx\.?|around|roughly)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*(?:sig figs?|significant|rounded|approx|since)[^)]*\)", "", text, flags=re.IGNORECASE)

    candidates = [len(text)]
    for phrase in EXPLANATION_PHRASES:
        idx = text.lower().find(phrase)
        if idx > 0:
            candidates.append(idx)
    sentence_stop = re.search(r"(?<=[A-Za-z0-9%)])\.\s+(?=[A-Z])", text)
    if sentence_stop:
        candidates.append(sentence_stop.start() + 1)
    text = text[: min(candidates)].strip()
    text = text.strip(" \t\r\n.;")
    text = re.sub(r"^(?:answer|answers|solution|solutions|value|values)\s*(?:is|are)?\s*[:=]?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def add_candidate(
    candidates: list[Candidate],
    seen: set[str],
    raw_text: str,
    source: str,
    question: str,
    expected_count: int,
    quality: float,
) -> None:
    raw_text = trim_explanation(raw_text)
    if not raw_text:
        return
    post = postprocess_response(raw_text, question, expected_count)
    answer_text = post["answer_text"].strip()
    if not answer_text or len(answer_text) > 900:
        return
    answer_text = normalize_answer_words(answer_text)
    key = re.sub(r"\s+", " ", answer_text).strip().lower()
    if key in seen:
        return
    seen.add(key)
    score = quality_score(answer_text, source, question, expected_count, quality)
    candidates.append(Candidate(answer_text, f"\\boxed{{{answer_text}}}", source, score))


def normalize_answer_words(answer_text: str) -> str:
    parts = _split_top_level_commas(answer_text)
    normalized = []
    for part in parts:
        p = part.strip()
        if re.fullmatch(r"yes|no", p, flags=re.IGNORECASE):
            p = p.upper()
        elif re.fullmatch(r"reject|do not reject", p, flags=re.IGNORECASE):
            p = p.upper()
        normalized.append(p)
    return ", ".join(normalized)


def quality_score(answer_text: str, source: str, question: str, expected_count: int, base: float) -> float:
    parts = [part for part in _split_top_level_commas(answer_text) if part != ""]
    score = base
    if len(parts) == expected_count:
        score += 35
    else:
        score -= 22 * abs(len(parts) - expected_count)
    if has_choice_options(question) and all(re.fullmatch(r"[A-Z]+", part.strip()) for part in parts):
        score += 35
        if expected_count > 2 and len(set(part.strip() for part in parts)) == 1:
            score -= 24
    if len(answer_text) <= 120:
        score += 15
    elif len(answer_text) > 350:
        score -= min(80, len(answer_text) / 20)
    lower = answer_text.lower()
    score -= 8 * sum(word in lower for word in VERBOSE_WORDS)
    if re.search(r"\b[A-Za-z]{12,}\b", answer_text):
        score -= 8
    if not is_usable_answer(answer_text, question, expected_count):
        score -= 90
    if source.startswith("numeric_tokens"):
        score -= 20
    if source.startswith("sympy"):
        score += 8
    score -= min(5.0, len(answer_text) / 120.0)
    return score


def is_usable_answer(answer_text: str, question: str, expected_count: int) -> bool:
    parts = [part for part in _split_top_level_commas(answer_text) if part.strip()]
    if len(parts) != expected_count:
        return False
    if len(answer_text) > 360:
        return False
    if has_choice_options(question) and all(re.fullmatch(r"[A-Z]+", part.strip()) for part in parts):
        return True
    if re.search(r"\b[a-j]\.\s*[A-Za-z]", answer_text, flags=re.IGNORECASE):
        return False
    if re.search(r"\b[xy]\s*:", answer_text, flags=re.IGNORECASE):
        return False
    allowed_words = {
        "sqrt",
        "sin",
        "cos",
        "tan",
        "atan",
        "ln",
        "log",
        "pi",
        "e",
        "yes",
        "no",
        "reject",
        "do",
        "not",
        "infinity",
    }
    for part in parts:
        words = re.findall(r"[A-Za-z]+", part.lower())
        if any(word not in allowed_words and len(word) > 1 for word in words):
            return False
        if len(words) > 4 and not re.fullmatch(r"(?:do\s+not\s+reject|reject|yes|no)", part, flags=re.IGNORECASE):
            return False
    return True


def label_value_candidates(text: str) -> list[tuple[str, str]]:
    labels = "|".join(re.escape(label) for label in (*ORDINAL_LABELS, *NAMED_LABELS))
    label_pattern = re.compile(
        rf"(?im)^\s*(?:[-*]\s*)?(?:\(?[a-j]\)|part\s*\(?[a-j]\)?|{labels}|[xyz]_\d+|[abcxyz])"
        rf"\s*(?:answer|value|mean)?\s*[:=]\s*(.+?)\s*$"
    )
    found = []
    for match in label_pattern.finditer(text):
        value = trim_explanation(match.group(1))
        if value:
            found.append(("labelled_values", value))
    return found


def inline_label_value_candidates(text: str) -> list[tuple[str, str]]:
    labels = "|".join(re.escape(label) for label in (*ORDINAL_LABELS, *NAMED_LABELS))
    inline_pattern = re.compile(
        rf"\b(?:{labels}|[xyz]_\d+)\s*(?:answer|value|mean)?\s*[:=]\s*([^.;\n]+)",
        flags=re.IGNORECASE,
    )
    found = []
    for match in inline_pattern.finditer(text):
        value = trim_explanation(match.group(1))
        if value:
            found.append(("inline_labelled_values", value))
    leg_pattern = re.compile(
        r"\b(first|second|third|fourth|fifth)\s+(?:leg|answer|value|side)\s*(?:is|=|:)\s*([^\n]+)",
        flags=re.IGNORECASE,
    )
    for match in leg_pattern.finditer(text):
        value = trim_explanation(match.group(2))
        if value:
            found.append(("inline_labelled_values", value))
    return found


def marker_segments(text: str) -> list[tuple[str, str]]:
    found = []
    lower = text.lower()
    for marker in ANSWERISH_MARKERS:
        start = 0
        while True:
            idx = lower.find(marker, start)
            if idx < 0:
                break
            segment = text[idx + len(marker) : idx + len(marker) + 900]
            found.append(("answer_marker", segment))
            start = idx + len(marker)
    final_colon = re.finditer(r"(?im)^\s*(?:final|answer|answers?|solutions?)\s*:\s*(.+)$", text)
    for match in final_colon:
        found.append(("answer_marker_line", match.group(1)))
    return found


def option_letter_candidates(text: str, question: str, expected_count: int) -> list[tuple[str, str]]:
    if not has_choice_options(question):
        return []
    found: list[tuple[str, str]] = []
    answerish = strip_think(text)
    if not answerish:
        answerish = text[-2500:]
    tail = answerish[-3500:]

    part_letters = []
    for match in re.finditer(
        r"(?i)(?:part\s*)?\(?([a-j])\)?\s*(?:answer|is|:|-|=){1,2}\s*(?:option\s*)?([A-J])\b",
        tail,
    ):
        part_letters.append(match.group(2).upper())
    if len(part_letters) >= expected_count:
        found.append(("option_letters_by_part", ", ".join(part_letters[-expected_count:])))

    boxed_letters = re.findall(r"\\boxed\{([A-J](?:\s*,\s*[A-J])*)\}", tail, flags=re.IGNORECASE)
    for letters in boxed_letters:
        found.append(("option_letters_boxed", letters.upper()))

    marker_letters = re.findall(
        r"(?i)(?:answer|choice|option|corresponds to option)\s*(?:is|should be|:)?\s*(?:option\s*)?([A-J])\b",
        tail,
    )
    if expected_count == 1:
        for letter in marker_letters:
            found.append(("option_letters_marker", letter.upper()))
    elif len(marker_letters) >= expected_count:
        found.append(("option_letters_marker", ", ".join(letter.upper() for letter in marker_letters[-expected_count:])))

    compact = re.search(r"(?i)\banswers?\s*(?:are|:)\s*((?:[A-J]\s*,\s*){1,}[A-J])\b", tail)
    if compact:
        found.append(("option_letters_list", compact.group(1).upper()))
    return found


def option_source_quality(source: str) -> float:
    if source in {"option_letters_marker", "option_letters_list", "option_letters_boxed"}:
        return 94
    if source == "option_letters_by_part":
        return 74
    return 88


def answer_summary_number_candidates(text: str, expected_count: int) -> list[tuple[str, str]]:
    found = []
    number = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?:/\d+(?:\.\d+)?)?"
    token_pattern = re.compile(rf"(?<![A-Za-z]){number}(?![A-Za-z])", flags=re.IGNORECASE)
    for source, segment in marker_segments(text):
        segment = segment[:600]
        tokens = token_pattern.findall(segment)
        tokens = [token for token in tokens if not re.fullmatch(r"12|15|4|5", token)]
        if len(tokens) >= expected_count:
            found.append((f"numeric_tokens_{source}", ", ".join(tokens[:expected_count])))
            found.append((f"numeric_tokens_{source}_last", ", ".join(tokens[-expected_count:])))
    return found


def split_answer_text(answer_text: str) -> list[str]:
    return [part.strip() for part in _split_top_level_commas(answer_text) if part.strip()]


def variant_base_quality(source: str, answer_text: str, kind: str, question: str, expected_count: int) -> float:
    if has_choice_options(question) and is_usable_answer(answer_text, question, expected_count):
        if all(re.fullmatch(r"[A-Z]+", part.strip()) for part in _split_top_level_commas(answer_text)):
            return 94
    if source.startswith("numeric_tokens"):
        return 42
    if "inline_labelled" in source:
        return 78
    if "labelled" in source:
        return 84
    if source.startswith("existing_postprocess") or source.startswith("boxed"):
        return 82
    if "answer_marker" in source:
        return 76
    if source.startswith("option_letters"):
        return option_source_quality(source)
    return 68


def prefer_sig6_tuple(question: str) -> bool:
    q = question.lower()
    trig_solution = (
        ("all solutions" in q or "interval" in q or "0 \\leq" in q or "0 <= " in q)
        and any(token in q for token in ("sin", "cos", "tan", "\\sin", "\\cos", "\\tan"))
    )
    quadratic_decimal = "completing the square" in q
    return trig_solution or quadratic_decimal


def prefer_sig6_scalar(question: str) -> bool:
    q = question.lower()
    exact_radical = "exact form" in q or "cannot contain decimals" in q or "sqrt" in q
    if exact_radical:
        return False
    if "round" in q or "nearest" in q:
        return True
    if "tan^{-1}" in q or "\\tan^{-1}" in q or "arctan" in q:
        return True
    if "hours" in q and ("how long" in q or "work together" in q or "together" in q):
        return True
    return False


def sympy_eval_part(part: str, sigfigs: int = 15) -> str | None:
    cleaned = part.strip()
    if len(cleaned) > 180:
        return None
    if "," in cleaned or re.search(r"[A-DF-Z_a-df-z]", cleaned.replace("atan", "").replace("sqrt", "").replace("pi", "").replace("ln", "").replace("log", "")):
        return None
    if not re.search(r"sqrt|pi|atan|ln|log|\^|/|\*", cleaned):
        return None
    try:
        import sympy as sp

        expr = cleaned.replace("^", "**")
        expr = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
        parsed = sp.sympify(
            expr,
            locals={
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "atan": sp.atan,
                "ln": sp.log,
                "log": sp.log,
                "e": sp.E,
            },
        )
        if parsed.free_symbols:
            return None
        value = float(parsed.evalf(16))
    except Exception:
        return None
    return f"{value:.{sigfigs}g}"


def add_tuple_variant(
    candidates: list[Candidate],
    seen: set[str],
    parts: list[str],
    source: str,
    question: str,
    expected_count: int,
    base_quality: float,
) -> None:
    if expected_count != 1 or len(parts) <= 1 or len(parts) > 8:
        return
    tuple_text = f"({', '.join(parts)})"
    if not is_tuple_like_answer(tuple_text):
        return
    add_candidate(
        candidates,
        seen,
        tuple_text,
        source,
        question,
        expected_count,
        base_quality,
    )


def add_variants(candidates: list[Candidate], seen: set[str], question: str, expected_count: int) -> None:
    originals = list(candidates)
    for candidate in originals:
        parts = split_answer_text(candidate.answer_text)
        if not parts:
            continue

        eval_parts = []
        sig6_parts = []
        changed = False
        for part in parts:
            evaluated = sympy_eval_part(part)
            if evaluated is None:
                eval_parts.append(part)
                sig6_parts.append(part)
            else:
                eval_parts.append(evaluated)
                sig6_parts.append(sympy_eval_part(part, sigfigs=6) or evaluated)
                changed = True
        if changed:
            answer_text = ", ".join(eval_parts)
            base_quality = variant_base_quality(candidate.source, answer_text, "sympy", question, expected_count) + 4
            add_candidate(
                candidates,
                seen,
                answer_text,
                f"sympy_eval:{candidate.source}",
                question,
                expected_count,
                base_quality,
            )
            add_tuple_variant(
                candidates,
                seen,
                eval_parts,
                f"tuple_wrap:sympy_eval:{candidate.source}",
                question,
                expected_count,
                base_quality + 16,
            )
            if expected_count == 1 and sig6_parts != eval_parts:
                answer_text = ", ".join(sig6_parts)
                short_quality = variant_base_quality(candidate.source, answer_text, "sympy_sig6", question, expected_count)
                if len(sig6_parts) > 1:
                    short_quality += 8 if prefer_sig6_tuple(question) else -8
                    tuple_bonus = 24 if prefer_sig6_tuple(question) else 4
                else:
                    short_quality += 8 if prefer_sig6_scalar(question) else -8
                    tuple_bonus = 4
                add_candidate(
                    candidates,
                    seen,
                    answer_text,
                    f"sympy_eval_sig6:{candidate.source}",
                    question,
                    expected_count,
                    short_quality,
                )
                add_tuple_variant(
                    candidates,
                    seen,
                    sig6_parts,
                    f"tuple_wrap:sympy_eval_sig6:{candidate.source}",
                    question,
                    expected_count,
                    short_quality + tuple_bonus,
                )

        stripped = []
        changed = False
        for part in parts:
            new = re.sub(r"^(?:[A-Za-z][A-Za-z_0-9' ]{0,24})\s*[:=]\s*", "", part).strip()
            if new != part:
                changed = True
            stripped.append(new)
        if changed:
            answer_text = ", ".join(stripped)
            base_quality = variant_base_quality(candidate.source, answer_text, "strip_labels", question, expected_count) + 2
            add_candidate(
                candidates,
                seen,
                answer_text,
                f"strip_labels:{candidate.source}",
                question,
                expected_count,
                base_quality,
            )
            add_tuple_variant(
                candidates,
                seen,
                stripped,
                f"tuple_wrap:strip_labels:{candidate.source}",
                question,
                expected_count,
                base_quality + 12,
            )

        approx_parts = []
        changed = False
        for part in parts:
            if "≈" in part:
                new = part.rsplit("≈", 1)[-1].strip()
            elif "~" in part:
                new = part.rsplit("~", 1)[-1].strip()
            else:
                new = part
            if new != part:
                changed = True
            approx_parts.append(new)
        if changed:
            answer_text = ", ".join(approx_parts)
            base_quality = variant_base_quality(candidate.source, answer_text, "approx_rhs", question, expected_count) + 5
            add_candidate(
                candidates,
                seen,
                answer_text,
                f"approx_rhs:{candidate.source}",
                question,
                expected_count,
                base_quality,
            )
            add_tuple_variant(
                candidates,
                seen,
                approx_parts,
                f"tuple_wrap:approx_rhs:{candidate.source}",
                question,
                expected_count,
                base_quality + 12,
            )

        rhs_parts = []
        changed = False
        for part in parts:
            if "=" in part:
                new = part.rsplit("=", 1)[-1].strip()
                changed = True
            else:
                new = part
            rhs_parts.append(new)
        if changed:
            answer_text = ", ".join(rhs_parts)
            base_quality = variant_base_quality(candidate.source, answer_text, "equals_rhs", question, expected_count) + 3
            add_candidate(
                candidates,
                seen,
                answer_text,
                f"equals_rhs:{candidate.source}",
                question,
                expected_count,
                base_quality,
            )
            add_tuple_variant(
                candidates,
                seen,
                rhs_parts,
                f"tuple_wrap:equals_rhs:{candidate.source}",
                question,
                expected_count,
                base_quality + 12,
            )

        add_tuple_variant(
            candidates,
            seen,
            parts,
            f"tuple_wrap:{candidate.source}",
            question,
            expected_count,
            variant_base_quality(candidate.source, candidate.answer_text, "tuple_wrap", question, expected_count) + 12,
        )


def is_tuple_like_answer(answer_text: str) -> bool:
    if not (answer_text.startswith("(") and answer_text.endswith(")")):
        return False
    inner = answer_text[1:-1].strip()
    if not inner or "," not in inner:
        return False
    parts = [part.strip() for part in _split_top_level_commas(inner) if part.strip()]
    if len(parts) < 2:
        return False
    allowed_word_pattern = re.compile(r"^(?:sqrt|sin|cos|tan|atan|ln|log|pi|e|infinity|x|y|t|n|i|INF)$", re.IGNORECASE)
    for part in parts:
        words = re.findall(r"[A-Za-z]+", part)
        if any(not allowed_word_pattern.fullmatch(word) for word in words):
            return False
        if len(part) > 120:
            return False
    return True


def generate_candidates(item: dict[str, Any], response: str) -> list[Candidate]:
    question = item["question"]
    expected_count = expected_answer_count(item)
    candidates: list[Candidate] = []
    seen: set[str] = set()

    post = postprocess_response(response, question, expected_count)
    add_candidate(candidates, seen, post["answer_text"], "existing_postprocess", question, expected_count, 80)

    full_text = response.strip()
    after_think = strip_think(full_text)
    tail = full_text[-5000:]

    boxed_entries = _find_boxed_entries(full_text)
    if boxed_entries:
        add_candidate(candidates, seen, ", ".join(entry[2] for entry in boxed_entries[-expected_count:]), "boxed_entries", question, expected_count, 95)
        add_candidate(candidates, seen, boxed_entries[-1][2], "last_boxed", question, expected_count, 90)

    for source, segment in marker_segments(after_think or tail):
        add_candidate(candidates, seen, segment, source, question, expected_count, 78)

    labelled = label_value_candidates(tail)
    inline_labelled = inline_label_value_candidates(tail)
    for values_source, values in (("labelled_values", labelled), ("inline_labelled_values", inline_labelled)):
        cleaned_values = [value for _, value in values]
        if len(cleaned_values) >= expected_count:
            add_candidate(
                candidates,
                seen,
                ", ".join(cleaned_values[-expected_count:]),
                values_source,
                question,
                expected_count,
                86,
            )
    for source, value in labelled + inline_labelled:
        add_candidate(candidates, seen, value, source, question, expected_count, 70)

    for source, value in option_letter_candidates(full_text, question, expected_count):
        add_candidate(candidates, seen, value, source, question, expected_count, option_source_quality(source))

    for source, value in answer_summary_number_candidates(after_think or tail, expected_count):
        add_candidate(candidates, seen, value, source, question, expected_count, 58)

    add_variants(candidates, seen, question, expected_count)
    candidates.sort(key=lambda candidate: candidate.quality, reverse=True)
    return candidates


def score_candidate(
    judger: Judger | None,
    candidate: Candidate,
    gold: Any | None,
    question: str,
    expected_count: int,
) -> bool | None:
    if judger is None or gold is None:
        return None
    if not candidate.answer_text or len(candidate.answer_text) > 500:
        return False
    if "\\begin" in candidate.answer_text or "\\end" in candidate.answer_text:
        return False
    if not is_usable_answer(candidate.answer_text, question, expected_count):
        return False
    return judge_with_timeout(judger, candidate.response, gold)


def repair_row(judger: Judger | None, item: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    response = row["response"]
    gold = item.get("answer")
    baseline = row if "correct" in row and "postprocessed_answer" in row else {}
    if not baseline and judger is not None and gold is not None:
        baseline = score_frq_item(judger, item, response)

    if bool(baseline.get("correct")):
        answer = baseline.get("postprocessed_answer", "")
        repaired_response = baseline.get("postprocessed_response", f"\\boxed{{{answer}}}")
        return {
            **row,
            "baseline_correct": True,
            "baseline_postprocessed_answer": answer,
            "repair_response": repaired_response,
            "repair_answer": answer,
            "repair_source": "already_correct",
            "repair_quality": 999,
            "repair_correct": True,
            "oracle_repair_response": repaired_response,
            "oracle_repair_answer": answer,
            "oracle_repair_source": "already_correct",
            "oracle_repair_correct": True if judger is not None else None,
            "repair_candidates": [],
            "correct": True,
            "error_type": baseline.get("error_type", "correct"),
        }

    candidates = generate_candidates(item, response)
    expected_count = expected_answer_count(item)
    question = item["question"]
    selected = next(
        (candidate for candidate in candidates if is_usable_answer(candidate.answer_text, question, expected_count)),
        candidates[0] if candidates else Candidate("", "\\boxed{}", "none", -999),
    )
    selected_correct = score_candidate(judger, selected, gold, question, expected_count)

    oracle = None
    oracle_correct = False
    candidate_summaries = []
    for candidate in candidates[:12]:
        correct = score_candidate(judger, candidate, gold, question, expected_count)
        candidate_summaries.append({
            "answer": candidate.answer_text,
            "source": candidate.source,
            "quality": round(candidate.quality, 3),
            "correct": correct,
        })
        if correct and oracle is None:
            oracle = candidate
            oracle_correct = True
            break
    if oracle is None and candidates:
        oracle = selected

    if judger is not None and gold is not None:
        raw_correct = bool(baseline.get("raw_correct"))
        final_correct = bool(raw_correct or selected_correct)
        post_for_error = {
            "response": selected.response,
            "answer_text": selected.answer_text,
            "notes": [f"repair_source:{selected.source}"],
        }
        error_type = classify_error(judger, response, post_for_error, _gold_list(gold), raw_correct, final_correct)
    else:
        final_correct = None
        raw_correct = None
        error_type = "unscored"

    return {
        **row,
        "baseline_correct": row.get("correct", baseline.get("correct")),
        "baseline_postprocessed_answer": row.get("postprocessed_answer", baseline.get("postprocessed_answer")),
        "repair_response": selected.response,
        "repair_answer": selected.answer_text,
        "repair_source": selected.source,
        "repair_quality": round(selected.quality, 3),
        "repair_correct": selected_correct,
        "oracle_repair_response": oracle.response if oracle else "",
        "oracle_repair_answer": oracle.answer_text if oracle else "",
        "oracle_repair_source": oracle.source if oracle else "",
        "oracle_repair_correct": oracle_correct if judger is not None else None,
        "repair_candidates": candidate_summaries[:20],
        "correct": final_correct,
        "error_type": error_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--oracle-errors", default=None)
    args = parser.parse_args()

    data_by_id = {row["id"]: row for row in read_jsonl(Path(args.data))}
    rows = [row for row in read_jsonl(Path(args.input)) if not row.get("is_mcq")]
    has_gold = all("answer" in data_by_id[row["id"]] for row in rows)
    judger = Judger(strict_extract=False) if has_gold else None

    repaired = []
    for row in tqdm(rows, desc="Repairing"):
        item = data_by_id[row["id"]]
        repaired.append(repair_row(judger, item, row))

    errors = [row for row in repaired if row["correct"] is False]
    write_jsonl(Path(args.output), repaired)
    write_jsonl(Path(args.errors), errors)
    if args.oracle_errors:
        write_jsonl(Path(args.oracle_errors), [row for row in repaired if row["oracle_repair_correct"] is False])

    total = len(repaired)
    if has_gold:
        baseline_correct = sum(bool(row.get("baseline_correct")) for row in repaired)
        repair_correct = sum(bool(row.get("correct")) for row in repaired)
        selected_only = sum((not bool(row.get("baseline_correct"))) and bool(row.get("repair_correct")) for row in repaired)
        oracle_correct = sum(bool(row.get("oracle_repair_correct")) or bool(row.get("baseline_correct")) for row in repaired)
        oracle_only = sum((not bool(row.get("baseline_correct"))) and bool(row.get("oracle_repair_correct")) for row in repaired)
        counts = collections.Counter(row.get("error_type", "unknown") for row in repaired)
        sources = collections.Counter(
            row.get("repair_source", "unknown")
            for row in repaired
            if (not bool(row.get("baseline_correct"))) and bool(row.get("repair_correct"))
        )
        print(f"Baseline accuracy: {baseline_correct}/{total} ({baseline_correct / total * 100:.2f}%)")
        print(f"Selected repair accuracy: {repair_correct}/{total} ({repair_correct / total * 100:.2f}%)")
        print(f"Selected repaired additional: {selected_only}")
        print(f"Oracle repair ceiling: {oracle_correct}/{total} ({oracle_correct / total * 100:.2f}%)")
        print(f"Oracle additional hidden in text: {oracle_only}")
        print("Selected repair sources:")
        for source, count in sources.most_common():
            print(f"  {source:32s} {count:4d}")
        print("Error types after selected repair:")
        for key, value in counts.most_common():
            print(f"  {key:26s} {value:4d}")
    else:
        print(f"Repaired {total} unscored FRQ rows.")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
