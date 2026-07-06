from __future__ import annotations

from difflib import SequenceMatcher

from querdex.extraction.models import AlignmentStatus

_MIN_FUZZY_RATIO = 0.75


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lowercase and collapse whitespace, keeping a map back to original indices.

    ``index_map[i]`` is the index in ``text`` of normalized character ``i``,
    which lets fuzzy matches report offsets in the original string.
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    prev_space = True  # drop leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            normalized_chars.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            normalized_chars.append(ch.lower())
            index_map.append(i)
            prev_space = False
    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        index_map.pop()
    return "".join(normalized_chars), index_map


def align(needle: str, haystack: str) -> tuple[int, int, AlignmentStatus] | None:
    """Locate needle inside haystack, returning (start, end, status) in haystack coordinates.

    Tries progressively looser strategies: exact, case-insensitive,
    whitespace-normalized, then fuzzy via difflib. Returns None when no
    sufficiently similar region exists - callers should treat that
    extraction as unverified model output.
    """
    needle = needle.strip()
    if not needle:
        return None

    idx = haystack.find(needle)
    if idx >= 0:
        return idx, idx + len(needle), "exact"

    idx = haystack.lower().find(needle.lower())
    if idx >= 0:
        return idx, idx + len(needle), "exact"

    norm_haystack, index_map = _normalize(haystack)
    norm_needle, _ = _normalize(needle)
    if not norm_needle or not norm_haystack:
        return None

    idx = norm_haystack.find(norm_needle)
    if idx >= 0:
        start = index_map[idx]
        end = index_map[idx + len(norm_needle) - 1] + 1
        return start, end, "fuzzy"

    matcher = SequenceMatcher(a=norm_haystack, b=norm_needle, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_haystack), 0, len(norm_needle))
    if match.size == 0:
        return None
    window_start = max(0, match.a - match.b)
    window_end = min(len(norm_haystack), window_start + len(norm_needle))
    if window_end <= window_start:
        return None
    window = norm_haystack[window_start:window_end]
    ratio = SequenceMatcher(a=window, b=norm_needle, autojunk=False).ratio()
    if ratio < _MIN_FUZZY_RATIO:
        return None
    start = index_map[window_start]
    end = index_map[window_end - 1] + 1
    return start, end, "fuzzy"
