"""Score a transcript against a reference text (WER + code-switch term recall).

Why this exists: the Day-1 "STT proven" claim rested on eyeballing one English
sentence, and it did not survive contact with Polish. Accuracy claims for this
project must be MEASURED against a HUMAN reference (D24) — a second machine decode is a
divergence baseline, not a reference. Deterministic — no LLM call (CLAUDE.md, Determinism First).

Usage:
    python scripts/score_transcript.py --reference scripts/inputs/test_script_pl.txt \
        --hypothesis "<transcript text>"
    python scripts/score_transcript.py --reference REF.txt --hypothesis-file HYP.txt

`--terms` scores recall of the English technical terms specifically — the
code-switch seam (G8) is where errors concentrate, so overall WER alone hides
the failure that matters most.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

# English technical fragments embedded in the Polish test script (G8 targets).
# Matched on the stem so Polish morphology ("fine-tuningu") still counts as a hit.
DEFAULT_TERMS: tuple[str, ...] = (
    "large language model",
    "fine-tuning",
    "lora",
    "trade-off",
    "pipeline",
    "retrieval augmented generation",
    "vector database",
    "chunking",
    "evaluation framework",
    "human review",
)


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation/diacritics-insensitive tokens for fair WER."""
    text = text.lower().replace("‑", "-")
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)


def wer(reference: list[str], hypothesis: list[str]) -> tuple[float, int, int, int]:
    """Word error rate via Levenshtein. Returns (wer, substitutions, deletions, insertions)."""
    n, m = len(reference), len(hypothesis)
    # dp[i][j] = (cost, sub, del, ins)
    prev: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub_c, sub_s, sub_d, sub_i = prev[j - 1]
            del_c, del_s, del_d, del_i = prev[j]
            ins_c, ins_s, ins_d, ins_i = cur[j - 1]
            best = min(
                (sub_c + 1, sub_s + 1, sub_d, sub_i),
                (del_c + 1, del_s, del_d + 1, del_i),
                (ins_c + 1, ins_s, ins_d, ins_i + 1),
            )
            cur.append(best)
        prev = cur
    cost, subs, dels, ins = prev[m]
    return (cost / n if n else 0.0), subs, dels, ins


def term_recall(hypothesis_text: str, terms: tuple[str, ...]) -> list[tuple[str, bool]]:
    """Did each English technical term survive transcription?

    Matches on TOKEN boundaries, with a Polish suffix allowed on the term's last
    token ("sourcingiem" counts for "sourcing", "frameworku" for "framework"). The
    earlier raw-substring test also fired on accidental matches inside unrelated
    Polish words, which a short term like "api" hits constantly.
    """
    hay = normalise(hypothesis_text)
    out = []
    for term in terms:
        stem = normalise(term)
        found = False
        if stem:
            head, last = stem[:-1], stem[-1]
            for i in range(len(hay) - len(stem) + 1):
                if hay[i : i + len(head)] == head and hay[i + len(head)].startswith(last):
                    found = True
                    break
        out.append((term, found))
    return out


# --- human-reference sample packs (G8 / #324) ---------------------------------

BLOCK_HEADER = re.compile(
    r"^===\s*(\d+)\s*\[(\d+:\d+)-(\d+:\d+)\]\s*(\S+)\s*\|\s*([a-z-]+)\s*\|"
)


def parse_sample_pack(path: Path) -> tuple[list[dict], tuple[str, ...]]:
    """Read a `diff_transcripts.py --sample` template the user has filled in.

    Returns (blocks, terms). A block carries the two machine transcripts and the
    human TRUE: line; blocks with an empty TRUE: are kept but skipped by the scorer,
    so a partially filled pack still scores what is there.
    """
    blocks: list[dict] = []
    terms: list[str] = []
    cur: dict | None = None
    field: str | None = None
    in_terms = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("=== TERMS"):
            in_terms = True
            field = None
            continue
        if in_terms:
            if line.startswith("==="):
                in_terms = False       # a later block header ends the term list
            elif line.startswith("#") or not line.strip():
                continue
            else:
                terms.append(line.strip().lstrip("-").strip())
                continue
        m = BLOCK_HEADER.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {
                "n": int(m.group(1)), "start": m.group(2), "end": m.group(3),
                "languages": m.group(4), "stratum": m.group(5),
                "live": "", "offline": "", "reference": "",
            }
            field = None
            continue
        if cur is None or line.startswith("#"):
            continue
        for key, tag in (("live", "LIVE:"), ("offline", "OFFLINE:"), ("reference", "TRUE:")):
            if line.startswith(tag):
                cur[key] = line[len(tag):].strip()
                field = key
                break
        else:
            if field and line.strip():  # continuation line
                cur[field] = (cur[field] + " " + line.strip()).strip()
    if cur:
        blocks.append(cur)
    return blocks, tuple(t for t in terms if t)


def _pooled_wer(rows: list[dict], key: str) -> tuple[float, int, int]:
    """Pool by summing edits and reference words — NOT by averaging per-window rates."""
    edits = ref_words = 0
    for b in rows:
        ref, hyp = normalise(b["reference"]), normalise(b[key])
        _, subs, dels, ins = wer(ref, hyp)
        edits += subs + dels + ins
        ref_words += len(ref)
    return (edits / ref_words if ref_words else 0.0), edits, ref_words


def score_sample_pack(path: Path, whole_live: str = "") -> int:
    """Score a filled sample pack: true WER per stratum, plus term recall.

    Strata are scored SEPARATELY on purpose. `worst-divergence` windows were picked
    because the two machine decodes disagreed most, so pooling them with the random
    controls would report a number biased upward by construction. The honest headline
    is the `random-control` stratum; the others are diagnostics.
    """
    blocks, pack_terms = parse_sample_pack(path)
    scored = [b for b in blocks if normalise(b["reference"])]
    print(f"sample pack     : {path.name}")
    print(f"windows         : {len(scored)} scored / {len(blocks)} in pack")
    if not scored:
        print("\nNothing scored: no TRUE: line is filled in yet. This pack still needs a human ear.")
        return 1

    by_stratum: dict[str, list[dict]] = {}
    for b in scored:
        by_stratum.setdefault(b["stratum"], []).append(b)

    print("\nTRUE WER by stratum (reference = the human TRUE: lines):")
    print(f"  {'stratum':<22} {'n':>3} {'ref words':>10} {'LIVE WER':>10} {'OFFLINE WER':>12}")
    for name in ("random-control", "english-stretch", "worst-divergence"):
        rows = by_stratum.get(name)
        if not rows:
            continue
        live_rate, _, ref_words = _pooled_wer(rows, "live")
        off_rate, _, _ = _pooled_wer(rows, "offline")
        print(f"  {name:<22} {len(rows):>3} {ref_words:>10} {live_rate:>9.1%} {off_rate:>11.1%}")
    live_rate, _, ref_words = _pooled_wer(scored, "live")
    off_rate, _, _ = _pooled_wer(scored, "offline")
    print(f"  {'ALL (stratified, not':<22} {len(scored):>3} {ref_words:>10} {live_rate:>9.1%} {off_rate:>11.1%}")
    print(f"  {' representative)':<22}")

    head = by_stratum.get("random-control")
    if head:
        rate, _, words = _pooled_wer(head, "live")
        se = (rate * (1 - rate) / words) ** 0.5 if words else 0.0
        print(
            f"\nHEADLINE (unbiased): live-pipeline WER {rate:.1%} +/- {1.96 * se:.1%} (95% CI, binomial)"
            f"\n  on {words} human-referenced words across {len(head)} randomly chosen windows."
        )

    terms = pack_terms or DEFAULT_TERMS
    truth = " ".join(b["reference"] for b in scored)
    live_text = " ".join(b["live"] for b in scored)
    spoken = [t for t, ok in term_recall(truth, terms) if ok]
    if spoken:
        got = dict(term_recall(live_text, tuple(spoken)))
        hits = sum(1 for t in spoken if got[t])
        print(f"\ncode-switch terms HUMAN-CONFIRMED in the scored windows: {hits}/{len(spoken)} survived")
        for t in spoken:
            print(f"  {'OK  ' if got[t] else 'LOST'}  {t}")
    rest = [t for t in terms if t not in spoken]
    if rest and whole_live:
        got = dict(term_recall(whole_live, tuple(rest)))
        hits = sum(1 for t in rest if got[t])
        print(
            f"\nremaining pack terms, scored against the WHOLE live transcript: {hits}/{len(rest)} present"
            "\n  (machine-corroborated only — these fall outside the human-referenced windows)"
        )
        for t in rest:
            print(f"  {'OK  ' if got[t] else 'LOST'}  {t}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--reference", help="path to reference text file")
    parser.add_argument(
        "--sample-pack",
        help="score a filled human-reference pack from diff_transcripts.py --sample "
             "(true WER per stratum + term recall) instead of a single reference file",
    )
    parser.add_argument("--hypothesis", help="transcript text (inline)")
    parser.add_argument("--hypothesis-file", help="path to transcript text file")
    parser.add_argument(
        "--terms", nargs="*", default=None, help="override the technical-term list"
    )
    parser.add_argument(
        "--terms-file",
        help="read the term list from a file (one per line, # comments) — e.g. a "
             "call-specific list, since DEFAULT_TERMS describes the scripted fixture only",
    )
    args = parser.parse_args()

    if args.sample_pack:
        whole_live = (
            Path(args.hypothesis_file).read_text(encoding="utf-8") if args.hypothesis_file else ""
        )
        sys.exit(score_sample_pack(Path(args.sample_pack), whole_live))

    if not args.hypothesis and not args.hypothesis_file:
        parser.error("give --hypothesis or --hypothesis-file")

    file_terms: tuple[str, ...] = ()
    if args.terms_file:
        file_terms = tuple(
            t.strip()
            for t in Path(args.terms_file).read_text(encoding="utf-8").splitlines()
            if t.strip() and not t.lstrip().startswith("#")
        )

    if not args.reference:
        # Term recall alone: no reference transcript exists, so WER is not computable
        # and is deliberately NOT printed rather than faked against another decode.
        if not (file_terms or args.terms):
            parser.error("give --reference, or --terms/--terms-file for a recall-only run")
        hyp_text = (
            Path(args.hypothesis_file).read_text(encoding="utf-8")
            if args.hypothesis_file
            else args.hypothesis or ""
        )
        terms = tuple(args.terms) if args.terms else file_terms
        results = term_recall(hyp_text, terms)
        hits = sum(1 for _, ok in results if ok)
        print("NO REFERENCE GIVEN — term recall only; WER is not computable and is not shown.")
        print(f"code-switch terms present in the hypothesis: {hits}/{len(results)}")
        for term, ok in results:
            print(f"  {'OK  ' if ok else 'LOST'}  {term}")
        sys.exit(0)

    ref_text = Path(args.reference).read_text(encoding="utf-8")
    hyp_text = (
        Path(args.hypothesis_file).read_text(encoding="utf-8")
        if args.hypothesis_file
        else args.hypothesis or ""
    )

    ref, hyp = normalise(ref_text), normalise(hyp_text)
    rate, subs, dels, ins = wer(ref, hyp)

    print(f"reference words : {len(ref)}")
    print(f"hypothesis words: {len(hyp)}")
    print(f"WER             : {rate:.1%}  (sub {subs}, del {dels}, ins {ins})")
    print(f"word accuracy   : {1 - rate:.1%}")

    terms = tuple(args.terms) if args.terms else (file_terms or DEFAULT_TERMS)
    results = term_recall(hyp_text, terms)
    hits = sum(1 for _, ok in results if ok)
    print(f"\ncode-switch terms (G8): {hits}/{len(results)} survived")
    for term, ok in results:
        print(f"  {'OK  ' if ok else 'LOST'}  {term}")

    sys.exit(0)


if __name__ == "__main__":
    main()
