"""The salience gate (D23): decide whether a detected question is worth a suggestion call.

Subtask #386. The gate sits **after** the deterministic question trigger (D20), never
instead of it: D20 answers "is this a question at all" for free, and only what survives
it reaches this module. On the real 42-minute HR screen D20 fired 24 times and roughly a
third of those were worth a model call — the rest were audio checks, agenda readouts and
the interviewer explaining the company. This module is what removes them.

**What the gate actually measures, and why it is not cosine.** The subtask row settled on
"embed the question, take the cosine against the bundle's plan steps and STAR topics, fire
above a threshold". That route was built and measured here on 2026-09-04, and it does not
work on this transcript — four variants, all in `--selftest-salience --backend embed`:

    nomic-embed-text, whole segment      range 0.498-0.575, ordering ~random
    nomic-embed-text, question sentences range 0.510-0.619, ordering ~random
    bge-m3,           whole segment      range 0.479-0.636, ordering ~random
    bge-m3,           question sentences range 0.370-0.599, ordering ~random

Two separate findings sit under that. First, `nomic-embed-text` has no usable Polish: on
clean hand-written Polish probes it got 1/7 top-1 against the very topics they were written
from, while the same probes in English got 4/4 at 0.57-0.81. `bge-m3` (multilingual) fixes
that — 5/7 on the same Polish probes. Second, and fatally, **fixing the model does not fix
the gate**, because topical relevance is not the thing being asked. An HR screen is
topically saturated: every turn is about the job, the team, the contract. The interviewer's
monologue about team integrations legitimately resembles "working with people outside your
own team", and it scored *above* every genuinely substantive question. The discriminating
property is a **speech act** — did the interviewer just ask the candidate to say something —
and no amount of topic similarity recovers it. The one number that settles it: across the 24
turns the salient ones average **0.510** and the non-salient ones **0.518**. The signal is not
weak, it is absent and very slightly inverted, so there is no threshold to pick — the best-F1
cut only reaches 0.55 by firing 21 of 24, which is not a gate.

So the shipped backend is the alternative the subtask row named for exactly this case: a
one-word YES/NO judgment. The measured comparison (24 labelled turns, 8 salient) is:

    backend                                        P     R    F1  fires  median  extra VRAM
    embed  bge-m3, best threshold 0.436          0.38  1.00  0.55  21/24    120ms      664 MB
    llm    llama3.2:3b, whole segment            0.67  0.50  0.57   6/24    134ms     2593 MB
    llm    llama3.2:3b, question sentences       0.88  0.88  0.88   8/24    123ms     2593 MB
    llm    llama3.1:8b, question sentences       0.62  1.00  0.76  13/24    142ms     5300 MB
    llm    interview-copilot:14b, q-sentences    0.80  1.00  0.89  10/24    157ms         0  <- CHOSEN

The 14B wins on the two axes that matter live. **Recall 1.00** — it drops none of the eight
substantive turns, and a missed suggestion mid-answer is the expensive error, not a wasted
call. And **zero extra VRAM**, because it is the model the suggestion path already keeps
resident: measured, a 3B judge beside the 14B leaves 2273 MiB for a Whisper that needs
~2200, which is not a margin to take into a live interview.

Read the P/R numbers with D20's caveat: the 24 turns are **in-sample**. The prompt was
chosen against them, so 0.89 is a regression guard, not an out-of-sample estimate.

Run it:
    python scripts/reasoning.py --selftest-salience                    # the shipped gate
    python scripts/reasoning.py --selftest-salience --salience-backend embed
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import settings
from scripts.llm_client import get_llm
from scripts.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import at runtime
    from scripts.reasoning import ContextBundle

logger = get_logger("salience")

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")

JUDGE_SYSTEM = """You judge ONE turn from a job interview, spoken by the INTERVIEWER and \
machine-transcribed (messy, may merge several sentences).
Answer YES only if in this turn the interviewer ASKS THE CANDIDATE TO SAY SOMETHING — a \
question or request the candidate is now expected to answer at length.
Answer NO if the turn is: an audio or connection check, small talk, the interviewer \
describing the agenda, the interviewer explaining the company/team/benefits/process, the \
interviewer answering the candidate, or a rhetorical or tag question.
Reply with exactly one word: YES or NO.

The candidate has prepared answers for these areas:
{plan}
A question landing in one of those areas is still YES; a monologue that merely mentions \
one is NO."""

JUDGE_USER = 'Interviewer turn:\n"{turn}"\n\nYES or NO?'


@dataclass(frozen=True)
class SalienceVerdict:
    """Why the gate let a segment through, or did not. `fire` is the only field the ambient
    loop acts on; the rest exists so #322's dashboard can show *why* a turn was dropped."""

    fire: bool
    reason: str
    backend: str
    latency_ms: float = 0.0
    score: float | None = None
    topic_id: str | None = None

    def log_line(self) -> str:
        score = "" if self.score is None else f" score={self.score:.3f}"
        topic = "" if self.topic_id is None else f" topic={self.topic_id}"
        return f"{self.backend}:{self.reason}{score}{topic} ({self.latency_ms:.0f} ms)"


def question_sentences(text: str, is_question: Callable[[str], bool] | None = None) -> str:
    """The interrogative part of a merged segment, or the whole thing if none stands out.

    This is the single biggest accuracy lever measured (0.57 -> 0.88 F1 on the 3B judge):
    `SEGMENT_MAX_SECONDS=30` merges a whole exchange into one line, so the real question
    arrives wrapped in greetings, backchannel and the interviewer's own monologue. Handing
    the judge only the interrogative sentences removes the distractors. Reuses D20's rule
    as the per-sentence predicate rather than inventing a second one.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text
    predicate = is_question or (lambda s: "?" in s)
    kept = [s for s in sentences if "?" in s or predicate(s)]
    return " ".join(kept) if kept else text


def bundle_topics(bundle: ContextBundle) -> list[tuple[str, str]]:
    """`(topic_id, text)` for every plan step and STAR entry — the corpus the embed backend
    scores against. Plan steps carry their key points, STAR entries their tags: the title
    alone is too short to embed usefully."""
    topics = [(f"plan:{s.id}", f"{s.title}. {' '.join(s.key_points)}".strip()) for s in bundle.plan]
    topics += [
        (f"star:{e.id}", f"{e.title}. {' '.join(e.tags)}".strip() if e.tags else e.title)
        for e in bundle.answer_bank
    ]
    return topics


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def embed(texts: Sequence[str], model: str | None = None, timeout: float | None = None) -> list[list[float]]:
    """Embeddings from the local Ollama endpoint. Raises on any transport failure — the
    caller decides whether that fails open (see `SalienceGate.evaluate`).

    Local-only by construction (SI1): the gate must never be the thing that ships a
    transcript off the box. There is no cloud embedding path here and adding one would be
    a new egress requiring its own security reconciliation.
    """
    import os

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    body = json.dumps({"model": model or settings.SALIENCE_EMBED_MODEL, "input": list(texts)}).encode()
    request = urllib.request.Request(
        f"{base}/embeddings",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    with urllib.request.urlopen(request, timeout=timeout or settings.SALIENCE_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    return [row["embedding"] for row in payload["data"]]


class SalienceGate:
    """Stateful gate: built once per session, asked once per surviving question.

    Construct it with the loaded bundle so the judge prompt names the areas the candidate
    actually prepared, and (for the embed backend) so the topic vectors are computed once
    instead of per question.
    """

    def __init__(
        self,
        bundle: ContextBundle,
        backend: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        embed_model: str | None = None,
        timeout: float | None = None,
        fail_open: bool | None = None,
        enabled: bool | None = None,
        is_question: Callable[[str], bool] | None = None,
    ) -> None:
        self.bundle = bundle
        self.backend = (backend or settings.SALIENCE_BACKEND).lower()
        self.model = model or settings.SALIENCE_MODEL or settings.LOCAL_MODEL
        self.threshold = settings.SALIENCE_THRESHOLD if threshold is None else threshold
        self.embed_model = embed_model or settings.SALIENCE_EMBED_MODEL
        self.timeout = settings.SALIENCE_TIMEOUT_SECONDS if timeout is None else timeout
        self.fail_open = settings.SALIENCE_FAIL_OPEN if fail_open is None else fail_open
        self.enabled = settings.SALIENCE_GATE_ENABLED if enabled is None else enabled
        self._is_question = is_question
        self._topics: list[tuple[str, str]] | None = None
        self._topic_vectors: list[list[float]] | None = None
        self._client = None

    @property
    def active(self) -> bool:
        return self.enabled and self.backend != "off"

    def describe(self) -> str:
        if not self.active:
            return "salience gate: OFF (every detected question fires)"
        if self.backend == "embed":
            return f"salience gate: embed {self.embed_model} >= {self.threshold:.2f}"
        return f"salience gate: {self.backend} {self.model} (fail {'open' if self.fail_open else 'closed'})"

    # -- backends ----------------------------------------------------------
    def _judge_system(self) -> str:
        plan = "\n".join(f"- {step.title}" for step in self.bundle.plan) or "- (no interview plan loaded)"
        return JUDGE_SYSTEM.format(plan=plan)

    def _ask_llm(self, turn: str) -> bool:
        """One-word YES/NO from the model the suggestion path already keeps resident.

        Goes through `llm_client.get_llm` rather than a second HTTP client so the gate
        inherits the local base URL and the `reasoning_effort="none"` guard — without it
        qwen3 burns hidden reasoning tokens to say one word (see llm_client.get_llm).
        `max_tokens=3` because the answer is one token and an overrun is pure latency.
        """
        if self._client is None:
            self._client = get_llm(backend="local", model=self.model, temperature=0.0, max_tokens=3)
        messages = [
            ("system", self._judge_system()),
            ("human", JUDGE_USER.format(turn=turn)),
        ]
        reply = self._client.invoke(messages, timeout=self.timeout)
        return str(reply.content).strip().upper().startswith("Y")

    def _score_embed(self, turn: str) -> tuple[float, str | None]:
        if self._topic_vectors is None:
            self._topics = bundle_topics(self.bundle)
            self._topic_vectors = embed([t for _, t in self._topics], self.embed_model, self.timeout)
        vector = embed([turn], self.embed_model, self.timeout)[0]
        best, best_id = 0.0, None
        for (topic_id, _), topic_vector in zip(self._topics or [], self._topic_vectors):
            score = cosine(vector, topic_vector)
            if score > best:
                best, best_id = score, topic_id
        return best, best_id

    # -- the one method the ambient loop calls -----------------------------
    def warm(self) -> None:
        """Pay the gate's start-up cost before the interview, not during it. For the embed
        backend that is the topic vectors; for the LLM backend it is the client + a model
        load, which is free when the suggestion model is already resident."""
        if not self.active:
            return
        try:
            if self.backend == "embed":
                self._score_embed("warm-up")
            else:
                self._ask_llm("warm-up")
        except Exception as error:  # noqa: BLE001 - warming is best-effort by design
            logger.warning("salience gate warm-up failed (will fail %s live): %s",
                           "open" if self.fail_open else "closed", error)

    def evaluate(self, text: str) -> SalienceVerdict:
        """Should this detected question reach the suggestion model?

        **Fails open by default** (`SALIENCE_FAIL_OPEN`): if the judge times out or the
        backend is unreachable, the segment fires and the copilot degrades to its
        pre-gate behaviour. A wasted call costs tokens; a suggestion the candidate needed
        and did not get costs the interview. Flip it only when calls are the scarce thing.
        """
        if not self.active:
            return SalienceVerdict(True, "gate-off", self.backend)
        turn = question_sentences(text, self._is_question)
        started = time.perf_counter()
        try:
            if self.backend == "embed":
                score, topic_id = self._score_embed(turn)
                elapsed = (time.perf_counter() - started) * 1000
                fire = score >= self.threshold
                return SalienceVerdict(
                    fire, "above-threshold" if fire else "below-threshold",
                    self.backend, elapsed, score, topic_id,
                )
            salient = self._ask_llm(turn)
            elapsed = (time.perf_counter() - started) * 1000
            return SalienceVerdict(salient, "salient" if salient else "not-salient", self.backend, elapsed)
        except Exception as error:  # noqa: BLE001 - any backend failure takes the fail-open path
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning("salience gate failed (%s) — failing %s: %s",
                           type(error).__name__, "open" if self.fail_open else "closed", error)
            return SalienceVerdict(self.fail_open, f"fail-{'open' if self.fail_open else 'closed'}",
                                   self.backend, elapsed)
