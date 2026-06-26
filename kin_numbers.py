"""
Kinyarwanda spoken-number → digit converter.

Converts word-form numbers in ASR output to digit form for display.
Used ONLY in transcribe.py (--digits flag). Never applied to training
data (pseudo_label.py) or evaluation (evaluate.py) — those pipelines
must keep word forms because the model was trained to produce them.

Handles:
  millions:   miliyoni N
  thousands:  igihumbi / ibihumbi N
  hundreds:   magana N, ijana (100)
  tens:       mirongo N, makumyabiri (20), icumi (10)
  units:      1–9 with all Bantu noun-class variants
  connectors: na, n'UNIT (fused form)

Examples:
  "igihumbi kimwe magana inani mirongo icyenda na gatandatu" → "1896"
  "miliyoni eshatu"                                          → "3000000"
  "mirongo itanu na gatanu"                                  → "55"
  "magana abiri na icumi n'umwe"                             → "211"

Conservative: only replaces sequences anchored by a scale word
(miliyoni / igihumbi / ibihumbi / magana / ijana / mirongo /
 icumi / makumyabiri). Bare unit words without a scale word are left
alone — they're usually adjectives, not standalone numbers.
"""

import re

# ── Lookup tables ──────────────────────────────────────────────────────────────
# Keys are lowercase. Multiple noun-class variants map to the same integer.

UNITS: dict[str, int] = {
    # 1
    "rimwe": 1, "umwe": 1, "imwe": 1, "kimwe": 1,
    # 2
    "kabiri": 2, "ibiri": 2, "abiri": 2, "bibiri": 2, "ebiri": 2, "zibiri": 2,
    # 3
    "gatatu": 3, "itatu": 3, "atatu": 3, "bitatu": 3, "eshatu": 3, "zitatu": 3,
    # 4
    "kane": 4, "ine": 4, "ane": 4, "bine": 4, "zine": 4,
    # 5
    "gatanu": 5, "itanu": 5, "atanu": 5, "bitanu": 5, "zitanu": 5,
    # 6
    "gatandatu": 6, "itandatu": 6, "atandatu": 6, "bitandatu": 6, "zitandatu": 6,
    # 7 — several noun-class agreement forms
    "irindwi": 7, "indwi": 7, "arindwi": 7, "birindwi": 7, "zirindwi": 7,
    # 8
    "umunani": 8, "inani": 8, "munani": 8,
    # 9
    "icyenda": 9,
}

SCALE_WORDS: set[str] = {
    "miliyoni", "igihumbi", "ibihumbi", "magana", "ijana",
    "mirongo", "icumi", "makumyabiri",
}

# ── Tokenizer ──────────────────────────────────────────────────────────────────

_unit_alt  = "|".join(sorted(UNITS.keys(),      key=len, reverse=True))
_scale_alt = "|".join(sorted(SCALE_WORDS,        key=len, reverse=True))

_TOK_RE = re.compile(
    rf"\b({_scale_alt}|{_unit_alt}|na)\b",
    re.IGNORECASE,
)

# Expand fused "n'UNITWORD" → "na UNITWORD" so the tokenizer sees two tokens.
# Only expands when the word after n' is a known unit word — safe to apply
# everywhere since it only fires on number vocabulary.
_NFUSE_RE = re.compile(rf"\bn'({_unit_alt})\b", re.IGNORECASE)


def _preprocess(text: str) -> str:
    return _NFUSE_RE.sub(r"na \1", text)


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse(tokens: list[str]) -> int | None:
    """
    Parse a flat, lowercase token list into an integer.
    Returns None if the list does not form a valid number expression.

    Grammar (each clause optional, must appear in this order):
      [N miliyoni] [N igihumbi/ibihumbi] [N magana | ijana]
      [N mirongo | makumyabiri | icumi] [na? UNIT]
    """
    total = 0
    i = 0
    n = len(tokens)

    def peek() -> str | None:
        return tokens[i] if i < n else None

    def take() -> str:
        nonlocal i
        val = tokens[i]
        i += 1
        return val

    def try_unit() -> int | None:
        nonlocal i
        t = peek()
        if t and t in UNITS:
            i += 1
            return UNITS[t]
        return None

    def skip_na():
        if peek() == "na":
            take()

    def try_hundreds_multiplier() -> int:
        """
        Parse the multiplier that follows miliyoni/igihumbi/ibihumbi.
        Without an explicit 'na' connector, the number that follows is
        multiplicative, not additive.
          ibihumbi eshatu       = 3 × 1000 = 3,000
          ibihumbi icumi        = 10 × 1000 = 10,000
          ibihumbi makumyabiri  = 20 × 1000 = 20,000
          ibihumbi magana abiri = 200 × 1000 = 200,000
        """
        nonlocal i
        # Simple unit 1–9 (e.g. eshatu = 3)
        u = try_unit()
        if u is not None:
            return u
        # 10 (icumi)
        if peek() == "icumi":
            take()
            return 10
        # 20 (makumyabiri)
        if peek() == "makumyabiri":
            take()
            return 20
        # Compound hundreds: magana N (e.g. magana arindwi = 700)
        if peek() == "magana":
            take()
            u = try_unit()
            if u is not None:
                return u * 100
        return 1   # default: igihumbi alone = 1,000

    # millions
    if peek() == "miliyoni":
        take()
        total += try_hundreds_multiplier() * 1_000_000
        skip_na()

    # thousands — multiplier can be a simple unit OR a compound (magana N)
    if peek() in ("igihumbi", "ibihumbi"):
        take()
        total += try_hundreds_multiplier() * 1_000
        skip_na()

    # hundreds
    if peek() == "ijana":
        take()
        total += 100
        skip_na()
    elif peek() == "magana":
        take()
        mul = try_unit()
        if mul is None:
            return None     # "magana" must be followed by a unit
        total += mul * 100
        skip_na()

    # tens
    if peek() == "mirongo":
        take()
        mul = try_unit()
        if mul is None:
            return None     # "mirongo" must be followed by a unit
        total += mul * 10
        skip_na()
    elif peek() == "makumyabiri":
        take()
        total += 20
        skip_na()
    elif peek() == "icumi":
        take()
        total += 10
        skip_na()

    # units
    u = try_unit()
    if u:
        total += u

    # must consume ALL tokens and produce a non-zero result
    if i != n or total == 0:
        return None
    return total


# ── Span finder ────────────────────────────────────────────────────────────────

def _find_spans(text: str):
    """
    Yield (start, end, digit_str) for number expressions in preprocessed text.
    Spans reference character positions in `text`.
    """
    toks: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(1).lower())
        for m in _TOK_RE.finditer(text)
    ]
    if not toks:
        return

    # Group adjacent tokens (only whitespace allowed between them).
    groups: list[list[tuple[int, int, str]]] = []
    cur = [toks[0]]
    for prev, curr in zip(toks, toks[1:]):
        if re.fullmatch(r"\s*", text[prev[1]:curr[0]]):
            cur.append(curr)
        else:
            groups.append(cur)
            cur = [curr]
    groups.append(cur)

    for group in groups:
        # Only attempt parse when group starts with a scale word.
        if group[0][2] not in SCALE_WORDS:
            continue
        # Try longest prefix first, shrink until parse succeeds.
        for end_idx in range(len(group), 0, -1):
            sub = group[:end_idx]
            value = _parse([t[2] for t in sub])
            if value is not None:
                yield sub[0][0], sub[-1][1], str(value)
                break


# ── Public API ─────────────────────────────────────────────────────────────────

def words_to_digits(text: str) -> str:
    """
    Replace Kinyarwanda number word sequences in `text` with digit strings.
    Non-number text is returned unchanged.
    """
    preprocessed = _preprocess(text)
    spans = list(_find_spans(preprocessed))
    if not spans:
        return preprocessed     # return with n' expansions fixed

    result = preprocessed
    for start, end, digits in reversed(spans):   # right-to-left preserves positions
        result = result[:start] + digits + result[end:]
    return result


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        # (input,                                                    expected)
        ("igihumbi kimwe magana inani mirongo icyenda na gatandatu", "1896"),
        ("miliyoni eshatu",                                          "3000000"),
        ("mirongo itanu na gatanu",                                  "55"),
        ("magana abiri na icumi n'umwe",                             "211"),
        ("ibihumbi bibiri na magana atanu",                          "2500"),
        ("ijana na makumyabiri na gatatu",                           "123"),
        ("icumi na kabiri",                                           "12"),
        ("makumyabiri",                                               "20"),
        ("ijana",                                                     "100"),
        # non-number text should be untouched
        ("petero na yohani bagiye mu rugo",                          "petero na yohani bagiye mu rugo"),
    ]
    passed = 0
    for inp, expected in cases:
        got = words_to_digits(inp)
        ok = got == expected
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {inp!r}")
        if not ok:
            print(f"       expected: {expected!r}")
            print(f"       got:      {got!r}")
        else:
            passed += 1
    print(f"\n{passed}/{len(cases)} passed")
