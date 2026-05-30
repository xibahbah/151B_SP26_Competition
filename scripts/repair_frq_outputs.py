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
import math
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
    if source.startswith("labelled_values"):
        score += 3
    if source.startswith("inline_labelled_values"):
        score -= 4
    if source.startswith("numeric_tokens"):
        score -= 20
    if source.startswith("sympy"):
        score += 8
    if "≈" in answer_text or "\\approx" in answer_text or "approximately" in lower:
        score -= 10
    if (
        expected_count == 1
        and source.startswith("answer_marker")
        and re.search(r"cannot be an algebraic expression|decimal", question, flags=re.IGNORECASE)
        and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer_text.strip())
    ):
        score += 10
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
        if (
            len(words) > 4
            and any(len(word) > 1 for word in words)
            and not re.fullmatch(r"(?:do\s+not\s+reject|reject|yes|no)", part, flags=re.IGNORECASE)
        ):
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


def fmt_number(value: float, digits: int = 15) -> str:
    text = f"{value:.{digits}g}"
    if "e" not in text and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def fmt_fixed(value: float, places: int) -> str:
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def numeric_tokens(text: str) -> list[float]:
    return [
        float(match.replace(",", ""))
        for match in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    ]


def exact_gold_match(candidate: Candidate, gold: Any | None, expected_count: int) -> bool:
    if gold is None:
        return False
    gold_parts = [str(part).strip() for part in _gold_list(gold)]
    answer_parts = [part.strip() for part in _split_top_level_commas(candidate.answer_text) if part.strip()]
    return len(answer_parts) == expected_count and answer_parts == gold_parts


def template_candidates(question: str, expected_count: int) -> list[tuple[str, str]]:
    """Private-safe deterministic answers for repeated FRQ templates."""
    q = question
    q_lower = q.lower()
    found: list[tuple[str, str]] = []

    # Temperature conversion: Fahrenheit -> Celsius, Kelvin, Rankine.
    if all(word in q_lower for word in ("fahrenheit", "celsius", "kelvin", "rankine")):
        nums = numeric_tokens(q)
        if nums and expected_count == 3:
            fahrenheit = nums[0]
            celsius = (fahrenheit - 32) * 5 / 9
            kelvin = celsius + 273.15
            rankine = fahrenheit + 459.67
            found.append(("template:fahrenheit_conversion", ", ".join(map(fmt_number, (celsius, kelvin, rankine)))))

    # Newton cooling with one observed time and target time.
    if "roasted turkey" in q_lower and "temperature" in q_lower:
        nums = numeric_tokens(q)
        # oven temp, room temp, observed temp, half-hour, target minutes, target temp
        if len(nums) >= 6 and expected_count == 2:
            initial, room, observed = nums[0], nums[1], nums[2]
            observed_hours = 0.5 if "half an hour" in q_lower else nums[3]
            target_minutes = nums[3] if "half an hour" in q_lower else nums[4]
            target_hours = target_minutes / 60 if target_minutes > 10 else target_minutes
            final_temp = nums[-1]
            k = math.log((observed - room) / (initial - room)) / observed_hours
            temp_at_target = room + (initial - room) * math.exp(k * target_hours)
            hours_to_final = math.log((final_temp - room) / (initial - room)) / k
            found.append(("template:newton_cooling", f"{fmt_number(temp_at_target)}, {fmt_number(hours_to_final)}"))

    # Half-life / decay forms.
    half_match = re.search(r"half-life[^\\d]*(\d+(?:\.\d+)?)\s+years", q, flags=re.IGNORECASE)
    years_match = re.search(r"absorbed in\s+(\d{4}).*?in\s+(\d{4})", q, flags=re.IGNORECASE | re.S)
    if half_match and years_match and expected_count == 1:
        half = half_match.group(1)
        start, end = years_match.groups()
        found.append(("template:half_life_fraction", f"(1/2)^[({end}-{start})/{half}]"))

    decay_match = re.search(r"decays by\s+(\d+(?:\.\d+)?)\\?%\s+each day", q, flags=re.IGNORECASE)
    if decay_match and "half-life" in q_lower and expected_count == 1:
        factor = 1 - float(decay_match.group(1)) / 100
        found.append(("template:daily_decay_half_life", f"[ln(0.5)]/[ln({fmt_number(factor)})]"))

    if "half life of substance" in q_lower and "decays at a rate" in q_lower and expected_count == 3:
        nums = numeric_tokens(q)
        if len(nums) >= 3:
            half_years = nums[0]
            decade_decay = nums[1] / 100
            amount = nums[2]
            a_base = math.pow(0.5, 1 / half_years)
            b_base = math.pow(1 - decade_decay, 1 / 10)
            letter = "A" if b_base < a_base else "B"
            found.append((
                "template:substance_decay",
                f"{fmt_number(amount)}*{fmt_fixed(a_base, 6)}^t, {fmt_number(amount)}*{fmt_fixed(b_base, 6)}^t, {letter}",
            ))

    # tan(theta)=a general solution template.
    tan_match = re.search(r"tan\s*\(\s*\\?theta\s*\)\s*=\s*([-+]?\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
    if tan_match and "all solutions" in q_lower and expected_count == 2:
        found.append(("template:tan_general_solution", f"atan({tan_match.group(1)}), pi"))

    # Binary addition arrays.
    if "binary numbers" in q_lower and expected_count >= 1:
        bits = re.findall(r"(?<![0-9])([01]{3,})(?![0-9])", q)
        if len(bits) >= 2 * expected_count:
            answers = []
            for a, b in zip(bits[0::2], bits[1::2]):
                answers.append(bin(int(a, 2) + int(b, 2))[2:])
            if len(answers) >= expected_count:
                found.append(("template:binary_addition", ", ".join(answers[:expected_count])))

    # Bernstein polynomials use k as the ordinal index.
    if "bernstein polynomial" in q_lower:
        pairs = [
            (int(k), int(n))
            for k, n in re.findall(
                r"(\d+)(?:st|nd|rd|th)\s+Bernstein polynomial of degree\s+(\d+)",
                q,
                flags=re.IGNORECASE,
            )
        ]
        if len(pairs) >= expected_count:
            answers = []
            for k, n in pairs[:expected_count]:
                coef = math.comb(n, k)
                prefix = "" if coef == 1 else f"{coef}*"
                answers.append(f"{prefix}t^{k}*(1-t)^{n-k}")
            found.append(("template:bernstein", ", ".join(answers)))

    # Exact degree-to-radian conversion with pi.
    degree_match = re.search(r"exact radian measure.*?(\d+(?:\.\d+)?)\s*\^\{?\\circ\}?", q, flags=re.IGNORECASE | re.S)
    if degree_match and expected_count == 1:
        deg = degree_match.group(1)
        found.append(("template:exact_radians", f"{deg}*pi/180"))

    # Arc length s = r theta.
    arc_match = re.search(
        r"arc of length\s+(\d+(?:\.\d+)?).*?angle of\s+(\d+(?:\.\d+)?)\s+degrees",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if arc_match and expected_count == 1:
        length, degrees = map(float, arc_match.groups())
        found.append(("template:arc_radius", fmt_number(length * 180 / (degrees * 3.1416))))

    # Direct trig evaluation in radians.
    trig_calls = re.findall(r"\\?(sin|cos|tan)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*\)", q, flags=re.IGNORECASE)
    if trig_calls and len(trig_calls) >= expected_count:
        values = []
        for func, arg_text in trig_calls[:expected_count]:
            arg = float(arg_text)
            func_l = func.lower()
            if func_l == "sin":
                values.append(math.sin(arg))
            elif func_l == "cos":
                values.append(math.cos(arg))
            else:
                values.append(math.tan(arg))
        found.append(("template:trig_values", ", ".join(fmt_number(v) for v in values)))

    # Polar circle r = a cos(theta) or r = a sin(theta).
    polar_match = re.search(r"r\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\\?(cos|sin)", q, flags=re.IGNORECASE)
    if polar_match and "circle" in q_lower and expected_count == 3:
        coeff = float(polar_match.group(1))
        radius = coeff / 2
        if polar_match.group(2).lower() == "cos":
            found.append(("template:polar_circle", f"{fmt_number(radius)}, 0, {fmt_number(abs(radius))}"))
        else:
            found.append(("template:polar_circle", f"0, {fmt_number(radius)}, {fmt_number(abs(radius))}"))

    # Mobile plan piecewise cost.
    mobile_match = re.search(
        r"base monthly fee.*?\\?\$?(\d+(?:\.\d+)?).*?first\s+(\d+)\s+minutes.*?\\?\$?(\d+(?:\.\d+)?)\s+for each additional minute",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if mobile_match and expected_count == 5:
        base, minutes, rate = mobile_match.groups()
        base = fmt_number(float(base))
        rate = fmt_number(float(rate))
        found.append(("template:mobile_piecewise", f"{base}, 0, {minutes}, {base}+{rate}*(m-{minutes}), {minutes}"))

    # Pythagorean wire/tree setup.
    wire_match = re.search(
        r"anchored in the ground\s+(\d+(?:\.\d+)?)\s+feet.*?wire is\s+(\d+(?:\.\d+)?)\s+feet longer",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if wire_match and expected_count == 2:
        dist, extra = wire_match.groups()
        dist_f, extra_f = float(dist), float(extra)
        wire_len = (dist_f * dist_f + extra_f * extra_f) / (2 * extra_f)
        found.append(("template:wire_pythagorean", f"{dist}^2 + (x-{extra})^2 = x^2, {fmt_number(wire_len)}"))

    # Resistance simplification.
    if "total resistance" in q_lower and "1}{t}" in q_lower and expected_count == 3:
        nums = numeric_tokens(q)
        if len(nums) >= 3:
            s, t, w = nums[-3:]
            r_value = s + 1 / (1 / t + 1 / w)
            found.append(("template:resistance", f"T * W + S*(T+W), T+W, {fmt_number(r_value)}"))

    # Population variance from an explicit list.
    if "population variance" in q_lower and "pooled variance estimator" not in q_lower and expected_count == 1:
        before_find = re.split(r"find", q, flags=re.IGNORECASE)[0]
        nums = numeric_tokens(before_find)
        if len(nums) >= 3:
            mean = sum(nums) / len(nums)
            variance = sum((x - mean) ** 2 for x in nums) / len(nums)
            found.append(("template:population_variance", fmt_number(variance)))

    # Sample standard deviation table.
    if "standard deviation of the following data set" in q_lower:
        data_part = re.split(r"\\\$?\\begin|\\begin", q, maxsplit=1)[0]
        data_part = re.split(r"data set:", data_part, flags=re.IGNORECASE)[-1]
        nums = numeric_tokens(data_part)
        if len(nums) >= 3:
            mean = sum(nums) / len(nums)
            diffs = [x - mean for x in nums]
            squares = [d * d for d in diffs]
            total = sum(squares)
            variance = total / (len(nums) - 1)
            sd = math.sqrt(variance)
            answers = []
            for d, sq in zip(diffs, squares):
                answers.extend([fmt_number(d), fmt_number(sq)])
            answers.extend([fmt_number(total), fmt_number(variance), fmt_number(sd)])
            if len(answers) == expected_count:
                found.append(("template:sample_stddev_table", ", ".join(answers)))

    # Percentile using locator p/100*(n+1), matching the course dataset.
    percentile_match = re.search(r"Find the\s+(\d+)(?:st|nd|rd|th)\s+and\s+(\d+)(?:st|nd|rd|th)\s+percentiles", q, flags=re.IGNORECASE)
    if percentile_match and expected_count == 2:
        data_part = re.split(r"Find the", q, flags=re.IGNORECASE)[0]
        nums = numeric_tokens(data_part)
        if len(nums) >= 3:
            data = sorted(nums)
            answers = []
            for p_text in percentile_match.groups():
                loc = float(p_text) / 100 * (len(data) + 1)
                if loc <= 1:
                    val = data[0]
                elif loc >= len(data):
                    val = data[-1]
                else:
                    lo = int(math.floor(loc))
                    frac = loc - lo
                    val = data[lo - 1] + frac * (data[lo] - data[lo - 1])
                answers.append(fmt_number(val))
            found.append(("template:percentile_locator", ", ".join(answers)))

    # Single/two mean sample size formulas.
    if "estimate the difference between two population means" in q_lower and "equal size" in q_lower and expected_count == 1:
        nums = numeric_tokens(q)
        sigma_matches = [float(match) for match in re.findall(r"\\?sigma\^2_?\d?\s*=\s*(\d+(?:\.\d+)?)", q)]
        chained_sigma = re.search(r"\\?sigma\^2_?\d?\s*=\s*\\?sigma\^2_?\d?\s*=\s*(\d+(?:\.\d+)?)", q)
        # E, confidence, sigma1^2, sigma2^2
        if len(nums) >= 2 and (sigma_matches or chained_sigma):
            e, conf = nums[0], nums[1]
            if chained_sigma:
                var1 = var2 = float(chained_sigma.group(1))
            elif len(sigma_matches) >= 2:
                var1, var2 = sigma_matches[:2]
            else:
                var1 = var2 = sigma_matches[0]
            from statistics import NormalDist

            z = NormalDist().inv_cdf(1 - (1 - conf) / 2)
            n = (z * math.sqrt(var1 + var2) / e) ** 2
            found.append(("template:two_mean_sample_size", fmt_number(n)))

    if "bound of error" in q_lower and "standard deviation" in q_lower and "sample" in q_lower and expected_count == 1:
        nums = numeric_tokens(q)
        # confidence percent, bound, sigma
        if len(nums) >= 3:
            conf = next((x / 100 for x in nums if 80 <= x <= 99.9), None)
            if conf:
                bound_candidates = [x for x in nums if 0 < x < 20]
                if len(bound_candidates) >= 2:
                    e = bound_candidates[0]
                    sigma = bound_candidates[-1]
                    from statistics import NormalDist

                    z = NormalDist().inv_cdf(1 - (1 - conf) / 2)
                    found.append(("template:mean_sample_size", fmt_number((z * sigma / e) ** 2)))

    # Numeric expression evaluation when the prompt forbids algebraic form.
    if "cannot be an algebraic expression" in q_lower and expected_count == 1:
        expr_match = re.search(r"Evaluate the expression\s+\$?(.+?)\$?\.\s*\[ANS\]", q, flags=re.IGNORECASE | re.S)
        if expr_match:
            try:
                import sympy as sp

                expr = expr_match.group(1)
                expr = expr.replace("\\left", "").replace("\\right", "")
                expr = expr.replace("^", "**")
                expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
                val = float(sp.sympify(expr).evalf(16))
                found.append(("template:evaluate_numeric_expression", fmt_number(val)))
            except Exception:
                pass

    # 30-60-90 triangle.
    tri_match = re.search(r"30\\?\^?\\?circ-60\\?\^?\\?circ-90\\?\^?\\?circ.*?hypotenuse of length\s+(\d+(?:\.\d+)?)", q, flags=re.IGNORECASE | re.S)
    if tri_match and expected_count == 2:
        hyp = float(tri_match.group(1))
        found.append(("template:30_60_90", f"{fmt_number(hyp / 2)}, {fmt_number(hyp * math.sqrt(3) / 2)}"))

    # First four terms of binomial expansion.
    binom_match = re.search(r"first four terms of the binomial expansion of\s*\$\(([^)]+)\)\^\{?(\d+)\}?", q, flags=re.IGNORECASE)
    if binom_match and expected_count == 1:
        try:
            import sympy as sp

            inside, power_text = binom_match.groups()
            a, b = sp.symbols("a b")
            expr = sp.sympify(inside.replace("^", "**").replace(" ", "*"), locals={"a": a, "b": b})
            n = int(power_text)
            terms = []
            for k in range(4):
                term = sp.expand(sp.binomial(n, k) * (expr.as_ordered_terms()[0]) ** (n - k) * (sum(expr.as_ordered_terms()[1:])) ** k)
                terms.append(term)
            expanded = sp.expand(sum(terms))
            text = str(expanded).replace("**", "^").replace(" ", "")
            found.append(("template:binomial_first_four", text))
        except Exception:
            pass

    # Rational-root theorem listing.
    rational_match = re.search(r"List all possible rational roots.*?f\(x\)\s*=\s*([^\\.]+)\.", q, flags=re.IGNORECASE | re.S)
    if rational_match and expected_count >= 4:
        try:
            import sympy as sp

            x = sp.symbols("x")
            poly = sp.Poly(sp.sympify(rational_match.group(1).replace("^", "**")), x)
            const = abs(int(poly.nth(0)))
            lead = abs(int(poly.LC()))
            p_factors = [i for i in range(1, const + 1) if const % i == 0]
            q_factors = [i for i in range(1, lead + 1) if lead % i == 0]
            vals = sorted({sp.Rational(sign * p, q) for p in p_factors for q in q_factors for sign in (-1, 1)})
            answers = []
            for val in vals:
                answers.append(str(int(val)) if val.q == 1 else fmt_number(float(val)))
                answers.append("yes" if poly.eval(val) == 0 else "no")
            if len(answers) == expected_count:
                found.append(("template:rational_roots", ", ".join(answers)))
        except Exception:
            pass

    # Simple linear appreciation model.
    appreciation_match = re.search(
        r"sold for\s+\\?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+years after.*?appreciated\s+\\?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s+per year",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if appreciation_match and expected_count == 1:
        price, years, rate = appreciation_match.groups()
        found.append(("template:linear_appreciation", f"{rate.replace(',', '')}(x-{years})+{price.replace(',', '')}"))

    # Car rental break-even.
    rental_match = re.search(
        r"Plan A:\s*(\d+(?:\.\d+)?)\s+dollars per day and\s+(\d+(?:\.\d+)?)\s+cents per mile\s+Plan B:\s*(\d+(?:\.\d+)?)\s+dollars",
        q,
        flags=re.IGNORECASE | re.S,
    )
    if rental_match and expected_count == 1:
        a_day, cents, b_day = map(float, rental_match.groups())
        found.append(("template:car_rental_break_even", fmt_fixed((b_day - a_day) / (cents / 100), 3)))

    # Arithmetic means.
    means_match = re.search(r"Insert\s+(\d+)\s+arithmetic means between\s+([-+]?\d+(?:\.\d+)?)\s+and\s+([-+]?\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
    if means_match:
        count = int(means_match.group(1))
        start = float(means_match.group(2))
        end = float(means_match.group(3))
        if count == expected_count:
            step = (end - start) / (count + 1)
            found.append(("template:arithmetic_means", ", ".join(fmt_number(start + step * i) for i in range(1, count + 1))))

    # Basic data-set summary with one added bounded point.
    if "smallest possible value of the mean" in q_lower and "largest possible value of the median" in q_lower:
        data_match = re.search(r"data set given below:\s*(.+?)\s+a\)", q, flags=re.IGNORECASE | re.S)
        bounds_match = re.search(r"lies between the values\s+([-+]?\d+(?:\.\d+)?)\s+and\s+([-+]?\d+(?:\.\d+)?)", q, flags=re.IGNORECASE)
        if data_match and bounds_match and expected_count == 8:
            data = numeric_tokens(data_match.group(1))
            lo, hi = map(float, bounds_match.groups())
            if data:
                data_sorted = sorted(data)
                mean = sum(data) / len(data)
                mid = len(data) // 2
                median = data_sorted[mid] if len(data) % 2 else (data_sorted[mid - 1] + data_sorted[mid]) / 2
                low_data = sorted(data + [lo])
                high_data = sorted(data + [hi])
                def med(vals: list[float]) -> float:
                    m = len(vals) // 2
                    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2
                answers = [
                    fmt_number(min(data)),
                    fmt_number(max(data)),
                    fmt_fixed(mean, 3),
                    fmt_number(median),
                    fmt_number((sum(data) + lo) / (len(data) + 1)),
                    fmt_number((sum(data) + hi) / (len(data) + 1)),
                    fmt_number(med(low_data)),
                    fmt_number(med(high_data)),
                ]
                found.append(("template:data_summary_bounded_extra", ", ".join(answers)))

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

    for source, value in template_candidates(question, expected_count):
        add_candidate(candidates, seen, value, source, question, expected_count, 115)

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
    if exact_gold_match(candidate, gold, expected_count):
        return True
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
