"""Source artifact inventory for Structured Add."""

from __future__ import annotations

import re

_FENCED_CODE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})(?P<language>[^\n]*)\n(?P<body>.*?)(?:\n(?P=fence))[ \t]*(?=\n|$)"
)
_DOLLAR_FORMULA = re.compile(r"(?ms)^\$\$[ \t]*\n?(?P<body>.*?)(?:\n?\$\$)[ \t]*(?=\n|$)")
_BRACKET_FORMULA = re.compile(r"(?ms)^\\\[[ \t]*\n?(?P<body>.*?)(?:\n?\\\])[ \t]*(?=\n|$)")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _overlaps(start: int, end: int, spans: list[tuple[int, int, dict[str, str]]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end, _ in spans)


def _line_spans(content: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    cursor = 0
    for line in content.splitlines(keepends=True):
        end = cursor + len(line)
        result.append((cursor, end, line.rstrip("\r\n")))
        cursor = end
    if cursor < len(content):
        result.append((cursor, len(content), content[cursor:]))
    return result


def inventory_source_artifacts(content: str) -> list[dict[str, str]]:
    """Return exact Markdown artifacts in source order with stable aliases."""

    spans: list[tuple[int, int, dict[str, str]]] = []
    for match in _FENCED_CODE.finditer(content):
        artifact = {
            "type": "code",
            "content": match.group("body"),
        }
        language = match.group("language").strip()
        if language:
            artifact["language"] = language
        spans.append((match.start(), match.end(), artifact))

    for pattern in (_DOLLAR_FORMULA, _BRACKET_FORMULA):
        for match in pattern.finditer(content):
            if not _overlaps(match.start(), match.end(), spans):
                spans.append((match.start(), match.end(), {"type": "formula", "content": match.group("body")}))

    lines = _line_spans(content)
    index = 0
    while index < len(lines):
        start, end, line = lines[index]
        if _overlaps(start, end, spans):
            index += 1
            continue
        if line.lstrip().startswith(">"):
            last = index
            while last + 1 < len(lines) and lines[last + 1][2].lstrip().startswith(">"):
                last += 1
            block_start, block_end = lines[index][0], lines[last][1]
            raw = content[block_start:block_end].rstrip("\r\n")
            spans.append((block_start, block_end, {"type": "quote", "content": raw}))
            index = last + 1
            continue
        if "|" in line and index + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[index + 1][2]):
            last = index + 1
            while last + 1 < len(lines) and "|" in lines[last + 1][2] and lines[last + 1][2].strip():
                last += 1
            block_start, block_end = lines[index][0], lines[last][1]
            raw = content[block_start:block_end].rstrip("\r\n")
            if not _overlaps(block_start, block_end, spans):
                spans.append((block_start, block_end, {"type": "table", "content": raw}))
            index = last + 1
            continue
        index += 1

    result: list[dict[str, str]] = []
    for artifact_index, (_, _, artifact) in enumerate(sorted(spans, key=lambda item: item[0]), start=1):
        result.append({"artifact_id": f"artifact-{artifact_index}", **artifact})
    return result
