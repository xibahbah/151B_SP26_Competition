#!/usr/bin/env python3
"""Analyze FRQ result/error files into actionable buckets."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "you", "your",
    "answer", "answers", "ans", "find", "what", "which", "using", "use", "given",
    "enter", "blank", "each", "where", "when", "then", "than", "into", "have",
    "has", "all", "one", "two", "three", "will", "value", "values", "following",
}


PATTERNS: list[tuple[str, list[str]]] = [
    ("embedded_choice", [r"\bA\.", r"\bB\.", r"\bC\.", r"choose", r"select", r"which of"]),
    ("multi_blank", [r"\[ANS\].*\[ANS\]"]),
    ("unit_conversion", [r"fahrenheit", r"celsius", r"kelvin", r"rankine", r"degrees", r"radians"]),
    ("linear", [r"slope", r"intercept", r"linear", r"line through", r"equation of.*line"]),
    ("sequence", [r"sequence", r"arithmetic", r"geometric", r"common ratio", r"common difference"]),
    ("trig", [r"sin", r"cos", r"tan", r"theta", r"trigonometric"]),
    ("polar", [r"polar", r"\br\s*=", r"theta"]),
    ("stats", [r"mean", r"median", r"standard deviation", r"z score", r"hypothesis", r"confidence"]),
    ("finance", [r"interest", r"annuity", r"loan", r"revenue", r"profit", r"cost function"]),
    ("exponential", [r"exponential", r"decay", r"growth", r"half-life", r"newton"]),
    ("calculus", [r"derivative", r"differentiate", r"integral", r"differential", r"tangent line"]),
    ("expression_format", [r"leave as", r"simplify", r"expression", r"formula", r"function"]),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def problem_tags(question: str) -> set[str]:
    lower = question.lower()
    tags = set()
    for tag, patterns in PATTERNS:
        if any(re.search(pattern, lower, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
            tags.add(tag)
    if question.count("[ANS]") <= 1:
        tags.add("single_blank")
    return tags


def words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z_'-]{2,}", text.lower())
        if token not in STOPWORDS
    ]


def snippet(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--examples", default=None, help="Optional JSONL file of grouped examples.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    data_by_id = {row["id"]: row for row in read_jsonl(Path(args.data))}
    rows = read_jsonl(Path(args.results))
    rows = [row for row in rows if not row.get("is_mcq")]

    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    print(f"FRQ rows: {total}")
    print(f"Correct: {correct}/{total} ({correct / total * 100:.2f}%)")
    print()

    error_counts = collections.Counter(row.get("error_type", "unknown") for row in rows)
    print("Error types:")
    for key, value in error_counts.most_common():
        print(f"  {key:26s} {value:4d}")
    print()

    tag_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    tag_totals: collections.Counter[str] = collections.Counter()
    word_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    grouped_examples: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)

    for row in rows:
        item = data_by_id.get(row["id"], {})
        question = item.get("question", "")
        err = row.get("error_type", "unknown")
        tags = problem_tags(question)
        for tag in tags:
            tag_totals[tag] += 1
            tag_counts[(err, tag)] += 1
            if not row.get("correct"):
                grouped_examples[(err, tag)].append(row)
        if not row.get("correct"):
            word_counts[err].update(words(question))

    print("Top error x pattern buckets:")
    for (err, tag), value in tag_counts.most_common(40):
        if err == "correct":
            continue
        print(f"  {err:24s} {tag:20s} {value:4d}")
    print()

    print("Top words by error type:")
    for err, counter in sorted(word_counts.items()):
        common = ", ".join(f"{word}:{count}" for word, count in counter.most_common(12))
        print(f"  {err:24s} {common}")
    print()

    print("Example failures:")
    for err, count in error_counts.most_common():
        if err == "correct":
            continue
        print(f"\n[{err}]")
        shown = 0
        for row in rows:
            if row.get("error_type") != err:
                continue
            item = data_by_id.get(row["id"], {})
            print(f"  id={row['id']} gold={row.get('gold')} pred={row.get('postprocessed_answer')}")
            print(f"    q: {snippet(item.get('question', ''))}")
            print(f"    r: {snippet(row.get('response', ''))}")
            shown += 1
            if shown >= args.limit:
                break

    if args.examples:
        out_path = Path(args.examples)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for (err, tag), examples in sorted(grouped_examples.items()):
                for row in examples[: args.limit]:
                    item = data_by_id.get(row["id"], {})
                    f.write(json.dumps({
                        "error_type": err,
                        "tag": tag,
                        "id": row["id"],
                        "gold": row.get("gold"),
                        "postprocessed_answer": row.get("postprocessed_answer"),
                        "question": item.get("question", ""),
                        "response": row.get("response", ""),
                    }) + "\n")
        print(f"\nWrote examples to {out_path}")


if __name__ == "__main__":
    main()
