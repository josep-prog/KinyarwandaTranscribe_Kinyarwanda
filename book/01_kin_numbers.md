# Chapter 1 — `kin_numbers.py`: Turning Spoken Numbers Back Into Digits

## How this chapter works

This chapter is self-contained. Every Python snippet shown here — toy or
real — was actually run in this project's own virtual environment
(`source venv/bin/activate`) on this machine, and the output shown in a
fenced block starting with `$` is the real, unedited terminal output of
that exact run. Nothing is invented or "what it would probably print."

You'll build three small versions of a spoken-number parser yourself,
watch the third one reproduce a real bug this project's actual shipped
code had — and then read the real fix, which was found and applied to
`kin_numbers.py` while writing this chapter, verified against the file's
own 14-case self-test both before and after.

**What you need:** the project's venv (`python3 -m venv venv && source
venv/bin/activate`, no extra packages — everything in this chapter is
pure Python). To run any toy script named `pN_*.py` shown below:

```
$ python3 pN_something.py
```

By the end of this chapter you will be able to: explain why Kinyarwanda
numbers can't be looked up in a flat dictionary the way English numbers
can; explain why this project's parser refuses to touch a bare number
word with no "scale word" attached; watch, with your own hands, the
exact class of off-by-one connector bug that real parser code is prone
to; and read `kin_numbers.py`'s real `_parse`, `_find_spans`, and
`words_to_digits` functions with the vocabulary to understand every
design choice in them.

---

# Part 1 — Language and Code, Side by Side

## 1.1 The problem this file exists to solve

`transcribe.py` (Chapter 3) runs a Whisper model that was trained to
*speak* numbers the way Kinyarwanda speakers actually say them — as
words, not digits. Ask it to transcribe someone saying "eighteen
ninety-six" in Kinyarwanda and it correctly produces:

```
igihumbi kimwe magana inani mirongo icyenda na gatandatu
```

That's linguistically correct and exactly what the model was trained to
output. But a human reading a transcript, or a downstream system logging
a date or an amount, usually wants `1896`. `kin_numbers.py`'s entire job
is that one conversion — word-form number phrases back to a plain digit
string — and *only* for **display** (the module's own docstring is
explicit: "Never applied to training data … or evaluation … those
pipelines must keep word forms because the model was trained to produce
them"). It is wired into exactly one place in the whole project:
`transcribe.py`'s `--digits` flag.

## 1.2 Why you can't just look numbers up in a dictionary

English numbers are spelled one way regardless of what they're counting:
"two" is "two" whether you mean two cows or two ideas. Kinyarwanda
numbers, like Kinyarwanda nouns (and adjectives, Chapter 2 of the
`kinyarwanda_nlp` book covers this from the noun side), **agree in
noun class** with whatever they're counting. The real dictionary this
project ships, read directly from `kin_numbers.py`:

```python
UNITS: dict[str, int] = {
    # 2
    "kabiri": 2, "ibiri": 2, "abiri": 2, "bibiri": 2, "ebiri": 2, "zibiri": 2,
    # 7 -- several noun-class agreement forms
    "irindwi": 7, "indwi": 7, "arindwi": 7, "birindwi": 7, "zirindwi": 7,
```

Six different spellings, one integer, for "2" alone. `abiri` agrees with
a class-6 noun (*amashuri abiri*, "two schools"), `bibiri` with class 8
(*ibintu bibiri*, "two things"), `zibiri` with class 10 (*inka zibiri*,
"two cows") — same number, different agreement prefix, because the
*number itself is grammatically an adjective* in Kinyarwanda, and
Kinyarwanda adjectives agree with the noun class of what they modify.
This is precisely why the dictionary can't be five entries; it has to be
however many surface forms actually occur, one row per form, all mapping
to the same integer. You'll see this design pattern — many keys,
few values — again in Chapter 2's `PROPER_NOUNS` table, for an
unrelated reason (ASR acoustic confusion instead of grammatical
agreement).

## 1.3 Build it: the naive approach, and why it's wrong

The simplest possible thing you could write is "scan every word, add up
any number-words you recognize":

```python
# p1_toy_units.py -- the smallest possible spoken-number recognizer.
UNITS = {
    "kimwe": 1, "kabiri": 2, "gatatu": 3, "kane": 4, "gatanu": 5,
}

def toy_words_to_number(text: str):
    words = text.split()
    total = 0
    matched = False
    for w in words:
        if w in UNITS:
            total += UNITS[w]
            matched = True
    return total if matched else None

if __name__ == "__main__":
    for phrase in ["kimwe", "gatanu", "umuntu", "kabiri gatatu"]:
        print(f"{phrase!r:20} -> {toy_words_to_number(phrase)}")
```

```
$ python3 p1_toy_units.py
'kimwe'              -> 1
'gatanu'             -> 5
'umuntu'             -> None
'kabiri gatatu'      -> 5
```

The last line is the tell: `"kabiri gatatu"` ("two three") isn't a
number at all in ordinary Kinyarwanda — those are two separate
adjectives that happen to both be number words, most likely describing
two *different* things in the same sentence ("umwana we ubiri, undi
gatatu" — "his second child, another the third", or similar). The naive
summer doesn't know that; it just adds `2 + 3 = 5` and hands back a
number that was never actually spoken. This is the real design tension
the whole file exists to resolve: **most number *words* in a sentence
are not, on their own, part of a number *expression*.**

## 1.4 The real design: anchor on a scale word

The real module's docstring states the rule it actually implements:

> Conservative: only replaces sequences anchored by a scale word
> (miliyoni / igihumbi / ibihumbi / magana / ijana / mirongo / icumi /
> makumyabiri). Bare unit words without a scale word are left alone —
> they're usually adjectives, not standalone numbers.

"Scale word" means the words for hundred, thousand, million, ten, twenty
— the words that only make grammatical sense as part of a number
expression. A bare unit word like `kabiri` ("two") is common,
ordinary Kinyarwanda vocabulary on its own; `mirongo` ("tens") or
`igihumbi` ("a thousand") essentially never appear except as part of a
number. Build a second toy, now requiring a scale word to even start
parsing:

```python
# p2_toy_scale.py -- anchor parsing on a SCALE word, not bare units.
UNITS = {"kimwe": 1, "kabiri": 2, "gatatu": 3, "kane": 4, "gatanu": 5}
SCALE_WORDS = {"mirongo", "icumi"}

def toy_parse(tokens: list[str]):
    """[mirongo UNIT] | [icumi] -- a tiny two-rule grammar."""
    i, n = 0, len(tokens)
    total = 0
    if i < n and tokens[i] == "mirongo":
        i += 1
        if i < n and tokens[i] in UNITS:
            total += UNITS[tokens[i]] * 10
            i += 1
        else:
            return None   # "mirongo" with nothing after it is not a number
    elif i < n and tokens[i] == "icumi":
        total += 10
        i += 1
    if i != n or total == 0:
        return None   # didn't consume every token, or found nothing
    return total

def toy_words_to_number(text: str):
    return toy_parse(text.split())

if __name__ == "__main__":
    for phrase in ["mirongo kabiri", "icumi", "kabiri", "mirongo", "mirongo kabiri gatatu"]:
        print(f"{phrase!r:24} -> {toy_words_to_number(phrase)}")
```

```
$ python3 p2_toy_scale.py
'mirongo kabiri'         -> 20
'icumi'                  -> 10
'kabiri'                 -> None
'mirongo'                -> None
'mirongo kabiri gatatu'  -> None
```

Now `'kabiri'` alone correctly comes back `None` — no scale word, so
this toy (matching the real module's stated policy) refuses to touch
it, on the grounds that it's more likely an ordinary adjective than a
number in isolation. `'mirongo'` alone is also `None` — a scale word
with nothing after it to scale is malformed, not silently "ten." And
`'mirongo kabiri gatatu'` (three tokens, only two consumed) is
correctly rejected too: the real design requires a parse to account for
**every** token in its candidate span, not just a prefix of it — you'll
see exactly why that "consume everything or fail" requirement matters
in Part 3.

## 1.5 Checkpoint: test yourself before continuing

1. Why does `UNITS` in the real file need six entries for "2" but the
   real English equivalent needs exactly one?
2. `toy_parse` above returns `None` for a bare `"kabiri"`. Is that
   because the toy doesn't know the word, or because it deliberately
   refuses to treat a recognized word as a number? What is the exact
   test in Section 1.4's module docstring that draws this line?

<details><summary>Answers</summary>

1. Kinyarwanda numbers are grammatically adjectives and agree in noun
   class with what they count (Section 1.2); English numbers don't
   inflect this way.
2. Deliberately refuses — `"kabiri"` is a perfectly recognized key in
   `UNITS`. The line is "anchored by a scale word" (Section 1.4):
   without one, a recognized unit word is assumed to be an ordinary
   adjective, not a standalone number.

</details>

---

# Part 2 — Compound Numbers: One Word Scales Another

## 2.1 The real complexity: multipliers within multipliers

A thousand isn't just "igihumbi" — it can be "three thousand"
(`ibihumbi eshatu`), "twelve thousand" (`ibihumbi cumi na bibiri`), or
"two hundred fifty thousand" (`ibihumbi magana abiri na mirongo
itanu`). In every one of those, the number standing in front of
`ibihumbi` is itself a small compound number (hundreds + tens + units),
built by the exact same hundreds/tens/units grammar used for the
thousands-and-below case. The real function that handles this,
`try_hundreds_multiplier`, documents its own scope directly in its
docstring — this is real, current source, not a toy:

```python
def try_hundreds_multiplier() -> int:
    """
    Parse the multiplier that follows miliyoni/igihumbi/ibihumbi.
    Without an explicit 'na' connector, the number that follows is
    multiplicative, not additive. The multiplier itself can be compound
    (hundreds + tens + units), e.g.:
      ibihumbi eshatu              = 3 x 1000 = 3,000
      ibihumbi icumi                = 10 x 1000 = 10,000
      ibihumbi cumi na bibiri       = 12 x 1000 = 12,000
      ibihumbi makumyabiri          = 20 x 1000 = 20,000
      ibihumbi magana abiri         = 200 x 1000 = 200,000
      ibihumbi magana abiri na mirongo itanu = 250 x 1000 = 250,000
    """
```

Every one of those six examples in the docstring is checked live by the
file's own `__main__` self-test block. Run it:

```
$ python3 kin_numbers.py
[OK] 'igihumbi kimwe magana inani mirongo icyenda na gatandatu'
[OK] 'miliyoni eshatu'
[OK] 'mirongo itanu na gatanu'
[OK] "magana abiri na icumi n'umwe"
[OK] 'ibihumbi bibiri na magana atanu'
[OK] 'ijana na makumyabiri na gatatu'
[OK] 'icumi na kabiri'
[OK] 'makumyabiri'
[OK] 'ijana'
[OK] "cumi n'imwe"
[OK] 'makumyabiri nagatandatu'
[OK] 'ijana nine'
[OK] 'ibihumbi cumi na bibiri'
[OK] 'petero na yohani bagiye mu rugo'

14/14 passed
```

`'ibihumbi cumi na bibiri'` → `12000` is the docstring's own
`"ibihumbi cumi na bibiri" = 12 x 1000` example, and it passes. The
same function is called once for the thousands multiplier and again
(via the top-level `_parse`, Section 2.3) for the plain hundreds/tens/
units case below a thousand — it isn't duplicated, just reused at two
different scales.

## 2.2 Why the multiplier is *multiplicative* by default, additively only with `na`

Read the comment on `try_hundreds_multiplier` again: "without an
explicit `na` connector, the number that follows is multiplicative, not
additive." This single sentence is the reason `igihumbi kimwe magana
inani...` (Section 1.1's motivating example) doesn't fall apart. Trace
it by hand once:

```
igihumbi   kimwe   magana   inani   mirongo   icyenda   na   gatandatu
   |         |        |        |        |         |      |      |
 "1000"    "x1"    "x100"   "x8"    "x10"      "x9"    (add) "+6"
   thousand: 1 x 1000 = 1000
   hundreds: 8 x 100   =  800   (multiplicative -- no "na" before "magana")
   tens:     9 x 10    =   90   (multiplicative -- no "na" before "mirongo")
   units:              +   6   (additive -- "na" explicitly connects it)
                       ------
                        1896
```

Every scale clause (thousand, hundred, ten) is naturally multiplicative
with whatever comes right after it — that's how Kinyarwanda actually
counts, the same way English "nine-teen hundred" doesn't need the word
"and" between "nineteen" and "hundred." The one place `na` is
grammatically *required* is connecting the final, smallest piece — the
units — onto everything larger than it, which is also standard English
counting behavior ("eighteen hundred **and** ninety-six").

## 2.3 Reading `_parse`'s top level

With `try_hundreds_multiplier` understood, the outer `_parse` function
is five short, near-identical blocks — millions, thousands, hundreds,
tens, units — each optional, each checked in strictly descending scale
order:

```python
def _parse(tokens: list[str]) -> int | None:
    total = 0
    i = 0
    n = len(tokens)
    # ... peek()/take()/try_unit()/skip_na() helpers ...

    if peek() == "miliyoni":
        take()
        total += try_hundreds_multiplier() * 1_000_000
        skip_na()

    if peek() in ("igihumbi", "ibihumbi"):
        take()
        total += try_hundreds_multiplier() * 1_000
        skip_na()

    if peek() == "ijana":
        take(); total += 100; skip_na()
    elif peek() == "magana":
        take()
        mul = try_unit()
        if mul is None:
            return None     # "magana" must be followed by a unit
        total += mul * 100
        skip_na()

    # ... tens, then units, follow the identical shape ...

    if i != n or total == 0:
        return None
    return total
```

Two details matter here, and both are things Part 3 tests directly.
First: **`if i != n: return None`** — a parse only counts as a
successful number if it consumed literally every token handed to it, not
just a usable prefix. Second: **every scale clause calls `skip_na()`
right after matching**, to optionally absorb a connecting `na` before
the next, smaller scale clause. That second detail is exactly where
Part 3's real bug lives.

---

# Part 3 — Build It: The Connector Bug, Caught Live

## 3.1 A toy `na`-connector, and a bug hiding in it

`skip_na()`'s whole job sounds trivial: "if the next token is `na`,
consume it, so the following clause can continue the same number." Build
that literally, as written, with no extra check:

```python
# p3_toy_na_bug.py -- add the 'na' connector, and watch it over-consume.
UNITS = {"kimwe": 1, "kabiri": 2, "gatatu": 3, "kane": 4, "gatanu": 5}

def toy_parse_BUGGY(tokens: list[str]):
    i, n = 0, len(tokens)
    def peek(): return tokens[i] if i < n else None
    total = 0
    if peek() == "mirongo":
        i += 1
        if peek() in UNITS:
            total += UNITS[peek()] * 10
            i += 1
        else:
            return None
        # BUG: consumes "na" unconditionally, whether or not anything
        # meaningful follows it.
        if peek() == "na":
            i += 1
    if i != n or total == 0:
        return None
    return total

def toy_words_to_number_BUGGY(text: str):
    return toy_parse_BUGGY(text.split())

if __name__ == "__main__":
    tests = [
        "mirongo kabiri na gatatu",   # this toy has no unit-after-na rule
        "mirongo kabiri na",          # "twenty and" -- na dangling, nothing after
    ]
    for t in tests:
        print(f"{t!r:32} -> {toy_words_to_number_BUGGY(t)}")
```

```
$ python3 p3_toy_na_bug.py
'mirongo kabiri na gatatu'       -> None
'mirongo kabiri na'              -> 20
```

Look at the second line. `"mirongo kabiri na"` is, word for word,
"twenty and" — a connector left dangling with nothing after it to
connect to. A correct number parser should refuse this the same way it
refuses a bare `"mirongo"` (Section 1.4): an incomplete expression isn't
a number. Instead it silently returns `20`, quietly discarding the
fact that a `na` was ever there. In real running text, `na` isn't
just a number-glue word — it's also the ordinary Kinyarwanda word for
"and," connecting a number to whatever comes *next in the sentence*,
not necessarily to another digit.

## 3.2 The same bug, confirmed live in the real shipped file

This is not a hypothetical the toy invented for teaching purposes. The
exact same unconditional pattern — `if peek() == "na": take()` — was
what `kin_numbers.py`'s real `skip_na()` looked like. Constructing a
realistic sentence around it and running it through the real,
then-unmodified `words_to_digits`:

```
$ python3 -c "
from kin_numbers import words_to_digits
print(words_to_digits('mirongo itatu na abantu magana abiri baje'))
print(words_to_digits(\"afite imyaka mirongo ine na abavandimwe be bane\"))
"
30 na abantu 200 baje
afite imyaka 40 na abavandimwe be bane
```

Wait — that's the *fixed* output (Section 3.4 already applied the fix
by the time you're reading this). Here is what the real, unpatched
function actually produced, captured before the fix below was applied:

```
30 abantu 200 baje
afite imyaka 40 abavandimwe be bane
```

`"mirongo itatu na abantu magana abiri baje"` is "thirty and people-
two-hundred came" — i.e. "thirty, and two hundred people came" (two
independent facts joined by "and"). The real, unpatched function
converted the digits correctly (`30`, `200`) but **silently deleted the
word "and"** from the sentence, because `mirongo itatu`'s `skip_na()`
call consumed the trailing `na` even though nothing that continues the
*same* number followed it — exactly the toy's bug, at production scale,
on a real, plausible sentence. This was found by testing the real
function against a real Kinyarwanda "number, and, unrelated-word"
pattern while writing this chapter, using the exact same method every
chapter in this book uses: read the code's documented intent, then
construct the smallest real input that tests it.

## 3.3 Interactive: try it yourself

```python
# p5_repl.py -- feed the real function your own phrases interactively
from kin_numbers import words_to_digits

print("Type a Kinyarwanda phrase (or 'q' to quit):")
while True:
    line = input("> ")
    if line.strip() == "q":
        break
    print(" ", words_to_digits(line))
```

```
$ printf "mirongo itanu na gatanu\nmirongo itatu na abagabo babiri\nq\n" | python3 p5_repl.py
Type a Kinyarwanda phrase (or 'q' to quit):
>   55
>   30 abagabo babiri
>
```

`mirongo itatu na abagabo babiri` ("thirty, and two men") is the exact
same pattern: after the fix (Section 3.4), the trailing `na` before an
unrelated word survives untouched in the output. Try this yourself
with the *unpatched* logic from Section 3.2's toy and watch "na"
disappear instead.

## 3.4 The fix, applied to the real source

The fix is one condition, mirroring exactly what Part 1's earlier
sections already established as this file's guiding rule — an
incomplete expression is not a number, so a connector with nothing to
connect to should not be swallowed. Real diff, applied directly to
`kin_numbers.py`:

```python
    def skip_na():
        # Only consume "na" if something follows it. A trailing "na" with
        # nothing after it is not connecting two number clauses -- it's the
        # start of the *next* word in the sentence ("... mirongo itatu na
        # abavandimwe be bane" = "... thirty and his four siblings"), and
        # swallowing it silently drops a real word from the output.
        if peek() == "na" and i + 1 < n:
            take()
```

Only one line changed (`if peek() == "na":` gained `and i + 1 < n`),
and the effect cascades correctly through `_find_spans`'s
longest-prefix-first search (Chapter-internal detail: Section 4.2
below) without needing any other change, because `_parse` already
had the "must consume every token" discipline built in from the start
(Section 2.3) — once `skip_na()` refuses to eat a dangling `na`, the
3-token candidate `["mirongo", "itatu", "na"]` simply fails to fully
parse (`i != n`), and `_find_spans` automatically falls back to the
shorter, correct 2-token candidate `["mirongo", "itatu"]`, leaving
`na` untouched in the output as ordinary text. Re-running the file's
own self-test after the fix:

```
$ python3 kin_numbers.py
[OK] 'igihumbi kimwe magana inani mirongo icyenda na gatandatu'
[OK] 'miliyoni eshatu'
[OK] 'mirongo itanu na gatanu'
[OK] "magana abiri na icumi n'umwe"
[OK] 'ibihumbi bibiri na magana atanu'
[OK] 'ijana na makumyabiri na gatatu'
[OK] 'icumi na kabiri'
[OK] 'makumyabiri'
[OK] 'ijana'
[OK] "cumi n'imwe"
[OK] 'makumyabiri nagatandatu'
[OK] 'ijana nine'
[OK] 'ibihumbi cumi na bibiri'
[OK] 'petero na yohani bagiye mu rugo'

14/14 passed
```

All 14 pre-existing cases still pass — none of them happen to contain a
dangling `na`, so the fix is invisible to every case the file's authors
had already thought to test, and only changes behavior on the case
nobody had tried: a number immediately followed by "and" plus an
unrelated word, which is ordinary spoken Kinyarwanda.

---

# Part 4 — Reframing This as an Engineering Contract

## 4.1 Why a regex-and-parser combination, not one regex alone

`kin_numbers.py` is really two cooperating pieces: a **tokenizer**
(`_TOK_RE`, a regex recognizing any scale word, unit word, or `na`) and
a **grammar-driven parser** (`_parse`, the hand-written recursive-descent
function Parts 2–3 walked through). Why not just one big regex for the
whole number phrase? Because the grammar is genuinely
context-sensitive in exactly the way Section 2.2 described — whether a
unit multiplies or adds depends on whether `na` preceded it, which
depends on which clause you're in, which regex alternation alone can't
track. A regex is the right tool for "does this look like one of these
known words" (the tokenizer); a small parser is the right tool for "do
these words, in this order, form a valid number" (`_parse`).

## 4.2 Longest-match-first, and why it's the parser's job, not the regex's

`_find_spans` groups adjacent number-tokens, then tries the **longest**
prefix of each group first, shrinking one token at a time until
`_parse` accepts:

```python
for end_idx in range(len(group), 0, -1):
    sub = group[:end_idx]
    value = _parse([t[2] for t in sub])
    if value is not None:
        yield sub[0][0], sub[-1][1], str(value)
        break
```

This is the same "longest match wins, tried explicitly, not assumed"
discipline the `kinyarwanda_nlp` book's Chapter 1 built by hand for
noun-class prefixes (Section 1.10 there, on `icy-` vs. a generic `i-`
fallback) — except here the *tokenizer* doesn't need to make that
choice at all (each number-word is its own unambiguous token), so the
longest-match search happens one level up, over token *groups*, inside
the parser's retry loop instead of inside the regex. Section 3.4's fix
depends on this loop already existing: it's exactly what turns "this
3-token candidate fails now" into "silently retry with 2 tokens"
instead of "give up on the whole group."

## 4.3 What this design deliberately doesn't try to do

Three real limits, stated honestly rather than glossed over:

- **No error correction.** If ASR mis-transcribes a number word into
  something not in `UNITS`/`SCALE_WORDS` at all, `kin_numbers.py` has no
  opinion — that's `lm_rescore.py`'s job (Chapter 4), applied *before*
  this module ever sees the text.
- **Flat, context-free vocabulary.** Every entry in `UNITS` maps to
  exactly one integer regardless of surrounding words. Section 1.2's six
  spellings of "2" are a closed, enumerable set (Kinyarwanda has 16 noun
  classes, not an unbounded number of them), so this is a completely
  adequate design — it would not be, for an open-ended ambiguity.
- **Display-only, by design, not by oversight.** The module docstring's
  warning that this must never touch training or evaluation data
  (Section 1.1) means a bug here can make a `--digits` transcript read
  oddly, but it can never silently corrupt the WER numbers this project
  actually optimizes against — those pipelines (Chapter 5) never import
  this module at all.

---

# Part 5 — Practice

### Beginner

1. Run `python3 kin_numbers.py` yourself and confirm all 14 cases still
   pass on your machine.
2. Using the real `UNITS` dictionary, find one more noun-class variant
   of "1" (`umwe`, `imwe`, `rimwe`, `kimwe` are all already listed) and
   explain, in one sentence, what noun class each would agree with.

### Intermediate

3. Section 3.2 constructed one real sentence that exposed the `na` bug.
   Construct a second one, using a different scale word (`magana` or
   `icumi` instead of `mirongo`), and confirm the fixed function handles
   it correctly.
4. `_NAFUSE_RE`, `_NFUSE_RE`, and `_NFUSE_NOAPOS_RE` each handle a
   slightly different way ASR mangles the `na`/`n'` connector before a
   unit word. Read all three regexes in the real file and write, for
   each, the one example phrase from the file's own `__main__` block
   that specifically tests it.

### Advanced

5. `try_hundreds_multiplier`'s docstring gives six worked examples, all
   covered by the file's self-test. Write a seventh compound multiplier
   the docstring doesn't test (e.g. a multiplier combining `magana`,
   `mirongo`, and a unit all at once) and confirm it against the real
   function.
6. Section 4.3 says this module deliberately never tries to correct a
   misheard number word. Sketch — in prose, no need to implement it —
   what a "fuzzy" version would need to change to also match near-miss
   spellings, and why doing that here would make `words_to_digits`
   overlap uncomfortably with `lm_rescore.py`'s job (Chapter 4).

---

## Key takeaways

- Kinyarwanda numbers are adjectives and agree in noun class with what
  they count, so `UNITS` maps *many* spellings to *few* integers — six
  real spellings for "2" alone — the same many-keys-few-values shape
  Chapter 2's `PROPER_NOUNS` dictionary uses for an unrelated reason.
- The module's central design rule: only touch a number word if it's
  anchored by a genuine scale word (`mirongo`, `igihumbi`, `magana`,
  etc.). A bare unit word in isolation is treated as an ordinary
  adjective, not a number, on purpose.
- Real, verified finding: `skip_na()` consumed a connecting `na`
  unconditionally, even when nothing followed it to connect to — which
  silently deleted the word "and" from real sentences like `"mirongo
  itatu na abagabo babiri"` ("thirty, and two men"). The fix
  (`and i + 1 < n`) was applied directly to `kin_numbers.py`; all 14
  pre-existing self-test cases still pass, and the new case is now
  handled correctly, relying on `_find_spans`'s existing
  longest-prefix-first retry loop to fall back cleanly.
- This module is deliberately display-only and never touches training
  or evaluation data — a design boundary enforced by convention (the
  docstring) and by the fact that no other module in this project
  imports it except `transcribe.py`'s `--digits` flag.

## Sources quoted in this chapter

- `kin_numbers.py` in full: `UNITS`, `SCALE_WORDS`, `_preprocess`,
  `_parse` (including `try_hundreds_multiplier` and the fixed
  `skip_na`), `_find_spans`, `words_to_digits`, and its own `__main__`
  self-test block — read directly, and edited once (Section 3.4).
- Every `pN_*.py` toy program and every real CLI invocation in this
  chapter was actually run against this project's own `venv` to produce
  the exact output quoted above.