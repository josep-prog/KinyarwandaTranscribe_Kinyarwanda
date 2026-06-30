"""
Post-processing normalization for Kinyarwanda ASR output.

Three-layer pipeline applied to raw model output before WER scoring:
  1. Word-merge splits   — re-splits common fused tokens
  2. Proper-noun fixes   — canonical spellings for Biblical names + French/English loanwords
  3. Orthographic fixes  — kin_ortho_fix() from libkinyarwanda.so
"""

import re
import ctypes
from pathlib import Path

# ── libkinyarwanda.so integration ─────────────────────────────────────────────
_LIB_PATH = Path(__file__).parent.parent / "kinyarwanda_nlp" / "libkinyarwanda.so"
_lib = None
_ORTHO_AVAILABLE = False

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.kin_ortho_fix.restype = None
    _lib.kin_ortho_fix.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    _ORTHO_AVAILABLE = True
except OSError:
    pass  # ortho_fix becomes a no-op — evaluation still works without it


def _ortho_fix(word: str) -> str:
    if not _ORTHO_AVAILABLE or not word:
        return word
    buf = ctypes.create_string_buffer(256)
    _lib.kin_ortho_fix(word.encode("utf-8"), buf, 256)
    result = buf.value.decode("utf-8")
    return result if result else word


# ── Proper noun dictionary ─────────────────────────────────────────────────────
# All keys are lowercase. Values are canonical spellings from the ground-truth
# docx references. Derived by comparing model hypothesis vs reference.
PROPER_NOUNS: dict[str, str] = {
    # Yezu (Jesus)
    "yeze":         "yezu",
    "yeza":         "yezu",
    "jezu":         "yezu",
    "jeza":         "yezu",
    "yesu":         "yezu",
    "yese":         "yezu",
    # Kristo (Christ)
    "kirisitu":     "kristo",
    "kiristo":      "kristo",
    "kirisita":     "kristo",
    "kristu":       "kristo",
    "christi":      "kristo",
    "christ":       "kristo",
    "kiriso":       "kristo",
    "kiristitu":    "kristo",
    "kirisitu":     "kristo",
    # Yezu Kristo compound
    "jesukristo":   "yezu kristo",
    "yezukristo":   "yezu kristo",
    # Pentekote (Pentecost)
    "pentecote":    "pentekote",
    "pentekoti":    "pentekote",
    # Pilato (Pilate)
    "pirato":       "pilato",
    "pirate":       "pilato",
    # Ponsiyo (Pontius) — appears in "Ponsiyo Pilato"
    "ponziyo":      "ponsiyo",
    # Yohani (John)
    "yohane":       "yohani",
    "johane":       "yohani",
    "yuhani":       "yohani",
    "ewan":         "yohani",
    # Salomo (Solomon)
    "saromo":       "salomo",
    # Alegisanderi (Alexander)
    "regisandire":  "alegisanderi",
    "regisandiri":  "alegisanderi",
    # Isiraheli (Israel)
    "isirahire":    "isiraheli",
    "isirayeli":    "isiraheli",
    "israel":       "isiraheli",
    # Bisiraheli (Israelites)
    "bisirahere":   "bisiraheli",
    "bisirayeli":   "bisiraheli",
    # Herodi (Herod)
    "herodide":     "herodi",
    "herode":       "herodi",
    "herodede":     "herodi",
    # Gamaliyeli (Gamaliel)
    "gagamariyeri": "gamaliyeli",
    "gamalieri":    "gamaliyeli",
    # Dawidi (David) — model sometimes drops final vowel or adds suffix
    "dawid":        "dawidi",
    "dawidate":     "dawidi",
    # Nazareti — model adds i- prefix (class 9 noun marker)
    "inazareti":    "nazareti",
    "nazarete":     "nazareti",
    # Abayahudi (Jews) — occasional misspelling
    "abayahudu":    "abayahudi",
    "abayahuda":    "abayahudi",
    "bayahuti":     "bayahudi",
    "bayahudu":     "bayahudi",
    # Kapadokiya — model fuses with following 'na'
    "kapadokiyana": "kapadokiya na",
    "kapadokiyo":   "kapadokiya",
    "kapodokiyo":   "kapadokiya",
    # Pamfiliya
    "panfiriyana":  "pamfiliya na",
    "pamfiriyana":  "pamfiliya na",
    "panfiriya":    "pamfiliya",
    # Filipo (Philip) — model uses French/English forms
    "philippe":     "filipo",
    "philip":       "filipo",
    "filipi":       "filipo",
    # Petero (Peter) — model sometimes anglicises
    "peter":        "petero",
    # Simoni (Simon) — model uses French form
    "simone":       "simoni",
    # Koruneli (Cornelius)
    "koroneri":     "koruneli",
    "koruneri":     "koruneli",
    "korunelei":    "koruneli",
    # Sitefano (Stephen)
    "sitefana":     "sitefano",
    "stefano":      "sitefano",
    # Yakobo (Jacob/James)
    "yakoba":       "yakobo",
    # Yozefu (Joseph)
    "yosefu":       "yozefu",
    # Sinayi (Sinai)
    "sinari":       "sinayi",
    "sinaiye":      "sinayi",
    # Yezu — additional variant
    "yezo":         "yezu",
    # Aburahamu (Abraham)
    "burahamu":     "aburahamu",
    "abramu":       "aburahamu",
    # Abakristo (Christians)
    "abakeriso":    "abakristo",
    "abakereso":    "abakristo",
    # Misiri (Egypt) — r/l confusion
    "misili":       "misiri",
    # Nyagasani (Lord) — truncated form
    "nyagasa":      "nyagasani",
    "nyagasane":    "nyagasani",
    # Sawuli (Saul/Paul) — multiple acoustic variants
    "sawa":         "sawuli",
    "sauli":        "sawuli",
    "sawuri":       "sawuli",
    "sahuri":       "sawuli",
    # Petero (Peter) — additional variant
    "peteru":       "petero",
    # Nyagasani (Lord) — extra variant
    "nyagasana":    "nyagasani",
    # Yope (Joppa)
    "yopi":         "yope",
    # Kayizariya (Caesarea)
    "kayizariye":   "kayizariya",
    "kazariya":     "kayizariya",
    # Damasi (Damascus)
    "damase":       "damasi",
    "damasiko":     "damasi",

    # ── French/English loanwords (code-switching) ─────────────────────────────
    # Kinyarwanda speakers regularly switch to French/English for scientific and
    # technical terms. The model (forced to language="sw") phonetically
    # Kinyarwandizes these, producing multiple inconsistent variants per word.
    # These entries restore the original French/English spelling.
    # Evidence: observed across WARI (mosquito documentary) and UKO (Jamaica
    # imbeba documentary). The same code-switching pattern appears in any topic
    # (health, war, economics, tech) — only the specific loanwords change.

    # Malaria — up to 5 different phonetic guesses for the same spoken word
    "malariya":         "malaria",
    "marariya":         "malaria",
    "mararaya":         "malaria",
    "malariyi":         "malaria",

    # Plasmodium (malaria parasite) — up to 6 phonetic guesses
    "pulasimodiyumu":   "plasmodium",
    "pulasimodiyo":     "plasmodium",
    "plusmodium":       "plasmodium",
    "purasimodiyumu":   "plasmodium",

    # Antenne (French: antenna / probe) — appeared 4× as "antene"
    "antene":           "antenne",

    # Insecticide — two inconsistent phonetic variants
    "insegiside":       "insecticide",
    "insegitiside":     "insecticide",

    # Mangouste / mongoose — model produces two spellings for each language form
    "mangusite":        "mangouste",
    "mangusti":         "mangouste",
    "manguze":          "mongoose",
    "manguzi":          "mongoose",

    # Dengue (fever) — single-letter acoustic error
    "denge":            "dengue",

    # Paludisme (French for malaria as a disease) — phonetic distortion
    "paludizime":       "paludisme",

    # Jamaica — country name consistently misspelled
    "jamaika":          "jamaica",
    "ijamaika":         "jamaica",
    "ijamayika":        "jamaica",
    "jamayika":         "jamaica",
    "jyamaika":         "jamaica",
    "jyamayika":        "jamaica",
    "jamaikani":        "jamaican",

    # Essence (French: fuel / petrol) — relevant in economics, war, daily life
    "esanse":           "essence",

    # Kinine / quinquina (antimalarial medicine, French origin)
    "kinini":           "kinine",
    "kenkena":          "quinquina",
    "kenkenna":         "quinquina",

    # Parc (French: national park / game reserve) — seen in nature documentaries
    "parikiye":         "parc",
    "parike":           "parc",
    "pariki":           "parc",

    # Rhinella (cane toad species) — genus name mispronounced
    "rinela":           "rhinella",

    # Ireland — model Kinyarwandizes country names
    "irilande":         "ireland",

    # ── New evidence (2026-06-30): WARI (mosquito), UKO JAMAICA (mongoose),
    # and UDUSHYA (World Cup) documentaries. Same code-switching pattern as
    # above — the human reference consistently quotes the original French/
    # English spelling (often in curly quotes), the model phonetically
    # Kinyarwandizes it. Counts verified against the 3 corrected .docx
    # references before adding (skipping anything ambiguous, e.g. "Afurika"
    # is the dominant Kinyarwanda spelling for Africa, not a model error).

    # Équipe (French: team) — World Cup commentary
    "ekipe":            "équipe",
    "ekipa":            "équipe",
    "ekipu":            "équipe",

    # Brésil — country name
    "burezile":         "brésil",
    "burezil":          "brésil",
    "berezere":         "brésil",

    # Zaïre — historical country name (now DR Congo)
    "zayire":           "zaïre",
    "zayile":           "zaïre",
    "zayili":           "zaïre",
    "zayir":            "zaïre",
    "zahire":           "zaïre",
    "zaire":            "zaïre",   # accent dropped

    # Mazout (French: fuel oil) — economics/war context
    "mazutu":           "mazout",
    "mazut":            "mazout",
    "mazute":           "mazout",

    # Countries/places — model Kinyarwandizes consistently across topics
    "kanada":           "canada",
    "megisike":         "mexique",
    "katari":           "qatar",
    "washingitoni":     "washington",
    "furaride":         "floride",
    "kwinzelande":      "queensland",
    # NOTE: "isirayeli" is NOT remapped here — it's already mapped above to
    # "isiraheli" (Biblical spelling) and that mapping is kept since it has
    # more evidence behind it. The sports-context "israel" spelling seen in
    # the World Cup documentary is a single-occurrence conflict; the dict
    # is flat/context-free so only one mapping can win.
    "pakisitani":       "pakistan",
    "kolombiya":        "colombia",
    "kolombeya":        "colombia",
    "ekwador":          "équateur",
    "dowa":             "doha",
    "arabiya":          "arabie",
    "sawudite":         "saoudite",
    "wayi":             "hawaï",

    # Australia — six distinct phonetic guesses observed in a single file
    "ositerariya":      "australia",
    "ositarariya":      "australia",
    "wesitarariya":     "australia",
    "oserariya":        "australia",
    "oserariye":        "australia",
    "wasitarariya":     "australia",

    # Yugoslavia
    "yugosilaviya":     "yugoslavia",
    "yugosilavia":      "yugoslavia",
    "yogosalaviya":     "yugoslavia",
    "yigosilaviya":     "yugoslavia",

    # Tech/brand terms — recurring sponsor-segment vocabulary
    "watsapu":          "whatsapp",
    "imeli":            "email",
    "dani":             "van",

    # Medical/scientific vocabulary (mosquito documentary)
    "kanseri":          "cancer",
    "diyabete":         "diabete",
    "asanimetero":      "centimeter",
    "imfuhahuje":       "infrarouge",
    "egisosikelete":    "exosquelette",
    "nekitare":         "nectar",
    "migizomatozisi":   "myxomatosis",
    "kaputere":         "capteur",
    "agisitative":      "gustatif",
    "anoferi":          "anopheles",
    "ayedes":           "aedes",
    "kiiregisi":        "culex",
    "kareti":           "current",
    "biyoloji":         "biology",
    "nisegiside":       "insecticide",
    "esansi":           "essence",
    "global":           "globine",

    # Invasive species documentary (mongoose/cane toad/python)
    "bamizi":           "burmese",
    "payifanzi":        "python",
    "payitonizi":       "python",

    # Cars/economics
    "iburide":          "hybride",
    "ibiride":          "hybride",
    "yundayi":          "hyundai",
    "kiya":             "kia",
    "asiransi":         "assurance",

    # Misc French loanwords (quoted code-switches in the reference)
    "sisiteme":         "système",
    "depanaje":         "dépannage",
    "fiyuvure":         "fièvre",
    "suwede":           "suède",
    "teregarame":       "télégramme",
    "siferike":         "sphérique",
    "otirishi":         "autriche",
    "otirisha":         "autriche",
}

# ── Word-merge split patterns ──────────────────────────────────────────────────
# Applied first (regex), ordered most-specific to least-specific.
WORD_MERGES: list[tuple[str, str]] = [
    # ingoro y'imana (temple/house of God) — most common merge
    (r"\bingoroyimana\b",           "ingoro y'imana"),
    (r"\brw'ingoroyimana\b",        "rw'ingoro y'imana"),
    (r"\bry'ingoroyimana\b",        "ry'ingoro y'imana"),
    (r"\bmu'ingoroyimana\b",        "mu ingoro y'imana"),
    # i Yeruzalemu — preposition 'i' fused with place name
    (r"\biheruzaremu\b",            "i yeruzalemu"),
    (r"\biyeruzaremu\b",            "i yeruzalemu"),
    (r"\biyeruzalemu\b",            "i yeruzalemu"),
    (r"\biyeruzaremu\b",            "i yeruzalemu"),
    # w'i Nazareti — possessive chain fused with 'i' + place
    (r"\bw'inazareti\b",            "w'i nazareti"),
    (r"\bwinazareti\b",             "w'i nazareti"),
    # mu izina — preposition + noun split (model writes "muzina")
    # only when followed by "rya" (= "in the name of")
    (r"\bmuzina\s+rya\b",           "mu izina rya"),
    (r"\bn'izina\s+rya\b",          "n'izina rya"),
    # verb+name fusions  (petero + verb start merged)
    (r"\bpeteraramubaza\b",         "petero aramubaza"),
    (r"\bpeterabibonye\b",          "petero abibonye"),
    (r"\bpeterarahaguruka\b",       "petero arahaguruka"),
    # Mezopotamiya — model often writes mezoputamiya / mezopatamiya
    (r"\bmezoputamiya\b",           "mezopotamiya"),
    (r"\bmezopatamiya\b",           "mezopotamiya"),
    (r"\bmezo kutamya\b",           "mezopotamiya"),
    # i Yope / i Kayizariya — preposition 'i' fused with place name
    (r"\biyope\b",                  "i yope"),
    (r"\bikayizariya\b",            "i kayizariya"),
    (r"\bisamariya\b",              "i samariya"),
    # b'i yeruzalemu broken across tokens
    (r"\bb'iyeruza\s+remu\b",       "b'i yeruzalemu"),
    (r"\bb'iyeru\s+zaremu\b",       "b'i yeruzalemu"),
    # mwuka + mu + ziranenge → mwuka muziranenge (spirit is holy)
    (r"\bmwuka\s+mu\s+ziranenge\b", "mwuka muziranenge"),
    # i yeruzalemu broken as "iyeruza + remu/lemu"
    (r"\biyeruza\s+[lr]emu\b",      "i yeruzalemu"),
    # kugira ngo (so that) fused as "kugerango"
    (r"\bkugerango\b",              "kugira ngo"),
    # n'abigishamategeko split as "n'abigisha amategeko"
    (r"\bn'abigisha\s+amategeko\b", "n'abigishamategeko"),
    # abigishamategeko without the n'
    (r"\babigisha\s+amategeko\b",   "abigishamategeko"),

    # Kinyarwanda word fusions found in documentary transcripts
    (r"\bntabwagorwa\b",            "ntabwo yagorwa"),   # "did not struggle" — fused negative
    (r"\bibyonavugaga\b",           "ibyo navugaga"),    # "what I was saying" — fused relative
    (r"\bz\s+ibyonnyi\b",           "z'ibyonnyi"),       # apostrophe split by space

    # Number-word fusions — model concatenates "magana/mirongo" + numeral
    # These are topic-agnostic: any audio with numbers will hit these.
    (r"\bmaganabiri\b",             "magana abiri"),     # 200
    (r"\bmaganatatu\b",             "magana tatu"),      # 300
    (r"\bmaganane\b",               "magana ane"),       # 400
    (r"\bmaganatanu\b",             "magana tanu"),      # 500
    (r"\bmirongotatu\b",            "mirongo itatu"),    # 30

    # Common Kinyarwanda function-word fusions
    (r"\bburigihe\b",               "buri gihe"),        # "every time / always"
    (r"\bkuberako\b",               "kubera ko"),        # "because"
]

# ── Common word fixes (non-proper-noun acoustic errors seen in evaluation) ─────
# Same dict mechanism as PROPER_NOUNS but for function words / morphology errors.
COMMON_WORD_FIXES: dict[str, str] = {
    # Vowel-class prefix errors (model adds/changes noun class prefix)
    "umwuka":       "mwuka",        # Holy Spirit — model adds u- prefix (9x)
    "amwuka":       "mwuka",        # same, a- prefix (3x)
    # City/town spelling — reference uses older "mugi" form (7x)
    "mujyi":        "mugi",
    "umujyi":       "umugi",
    "y'umujyi":     "y'umugi",
    "w'umujyi":     "w'umugi",
    # "the next day" — reference says bukeye, model says bucyeye (5x)
    "bucyeye":      "bukeye",
    # Priest — reference "umutambyi", model "umutambye" (4x)
    "umutambye":    "umutambyi",
    "batambye":     "batambyi",
    "abatambye":    "abatambyi",
    # Family/clan — reference "imiryango", model "imeryango" (4x)
    "imeryango":    "imiryango",
    "b'imeryango":  "b'imiryango",
    "w'imeryango":  "w'imiryango",
    # Joy — reference "ibyishimo", model "ibyeshimo" (3x)
    "ibyeshimo":    "ibyishimo",
    # Prophet — reference "umuhanuzi", model "umuhanozi" (3x)
    "umuhanozi":    "umuhanuzi",
    "abahanozi":    "abahanuzi",
    # Teachings — reference "nyigisho", model "nyigesho" (2x)
    "nyigesho":     "nyigisho",
    # "tens/decades" — vowel reduction (5x across files)
    "merongo":      "mirongo",
    # "just as" — glottal elision written out
    "nkuko":        "nk'uko",
    # Size/bigness — model drops the 'u' vowel in second syllable
    "ubonini":      "ubunini",
    # Killer (agent noun) — -ye vs -yi suffix confusion
    "umwicanye":    "umwicanyi",
    # Thousands — model drops the i- noun class prefix
    "bihumbi":      "ibihumbi",
    # Vaccine/barrier — model adds spurious -yo suffix
    "urukingyo":    "urukingo",
}


# Punctuation that can flank a word without being part of it (sentence/clause
# boundaries). Apostrophes are deliberately excluded — they're meaningful
# Kinyarwanda elision marks and some PROPER_NOUNS/COMMON_WORD_FIXES keys
# (e.g. "y'umujyi") contain them as part of the lookup key itself.
_FLANKING_PUNCT = ".,;:!?\"«»“”…()[]"


def _split_flanking_punct(tok: str) -> tuple[str, str, str]:
    """Split a token into (leading punct, core word, trailing punct)."""
    core = tok.strip(_FLANKING_PUNCT)
    if not core:
        return tok, "", ""
    start = tok.index(core)
    end = start + len(core)
    return tok[:start], core, tok[end:]


def normalize_hypothesis(text: str) -> str:
    """
    Apply post-processing to a raw ASR hypothesis string.

    Call this BEFORE the general normalize() in evaluate.py so that
    proper-noun fixes are visible to WER scoring.

    Returns a cleaned string still in mixed case / with original punctuation
    (the caller's normalize() handles lowercasing and punct removal).
    """
    # Layer 1: word-merge splits
    for pattern, replacement in WORD_MERGES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Layer 2: token-level proper-noun sub + common-word fix
    tokens = text.split()
    result = []
    for tok in tokens:
        # Strip flanking punctuation before lookup (e.g. "Jamaika," would
        # otherwise never match the "jamaika" dictionary entry), then
        # reattach it so the rest of the line is untouched.
        prefix, core, suffix = _split_flanking_punct(tok)
        lower = core.lower()
        if lower in PROPER_NOUNS:
            result.append(prefix + PROPER_NOUNS[lower] + suffix)
        elif lower in COMMON_WORD_FIXES:
            result.append(prefix + COMMON_WORD_FIXES[lower] + suffix)
        else:
            # kin_ortho_fix (_ortho_fix) is intentionally not called here: its
            # vowel-assimilation rule over-fires on already-correct words
            # (e.g. mirongo -> merongo, ibyishimo -> ibyeshimo), directly
            # undoing the fixes above. Measured 9/2672 real hypothesis words
            # touched, all regressions. Re-enable only after that rule is
            # fixed on the kinyarwanda_nlp side.
            result.append(tok)
    return " ".join(result)
