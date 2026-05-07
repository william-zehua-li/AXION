import re


# Words that signal low-confidence content — lower the confidence score.
_UNCERTAINTY_WORDS = frozenset({
    "maybe", "perhaps", "possibly", "unclear", "uncertain", "might",
    "could", "unsure", "unknown", "approximately", "roughly", "seems",
})


def _extract_candidates(loop_outputs: list, final_answer: str) -> list[str]:
    """Pull candidate knowledge strings from loop outputs and the final answer."""
    candidates: list[str] = []

    for tick_out in loop_outputs:
        # RAG ticks may carry retrieved chunks that the model synthesised new info from.
        for chunk in tick_out.get("rag_chunks", []):
            if isinstance(chunk, str) and chunk.strip():
                candidates.append(chunk.strip())
        # Tool outputs may contain facts worth persisting.
        tool_out = tick_out.get("tool_output")
        if isinstance(tool_out, str) and tool_out.strip():
            candidates.append(tool_out.strip())

    # The final answer itself is always a candidate.
    if final_answer and final_answer.strip():
        candidates.append(final_answer.strip())

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


class UpdateGate:
    def __init__(self, file_store, novelty_threshold: float = 0.5,
                 confidence_threshold: float = 0.7):
        self.file_store           = file_store
        self.novelty_threshold    = novelty_threshold
        self.confidence_threshold = confidence_threshold

    def evaluate(self, loop_outputs: list, final_answer: str) -> dict:
        written:  list[str] = []
        rejected: list[str] = []
        queued:   list[str] = []

        candidates = _extract_candidates(loop_outputs, final_answer)

        for candidate in candidates:
            novelty, confidence = self._score(candidate)

            if novelty < self.novelty_threshold:
                # Redundant — already know this.
                rejected.append(candidate)
            elif confidence >= self.confidence_threshold:
                # Novel and reliable — write immediately.
                self.file_store.append(candidate)
                written.append(candidate)
            else:
                # Novel but uncertain — queue for human review.
                self.file_store.queue(candidate)
                queued.append(candidate)

        return {"written": written, "rejected": rejected, "queued": queued}

    def _score(self, candidate: str) -> tuple[float, float]:
        """
        Returns (novelty, confidence) both in [0, 1].

        Novelty: 1 - max_cosine_similarity against existing store chunks.
          High score → the candidate contains information not yet in the store.

        Confidence: heuristic on text quality and absence of uncertainty markers.
        """
        novelty    = self._novelty_score(candidate)
        confidence = self._confidence_score(candidate)
        return novelty, confidence

    def _novelty_score(self, candidate: str) -> float:
        similar = self.file_store.retrieve(candidate, top_k=1)
        if not similar:
            return 1.0  # store is empty — everything is novel

        # Use the file store's own TF-IDF index via a small similarity check.
        # We get the top-1 result; estimate overlap by token Jaccard similarity.
        top_chunk   = similar[0]
        cand_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        top_tokens  = set(re.findall(r"[a-z0-9]+", top_chunk.lower()))
        if not cand_tokens or not top_tokens:
            return 1.0
        intersection = cand_tokens & top_tokens
        union        = cand_tokens | top_tokens
        jaccard      = len(intersection) / len(union)
        return 1.0 - jaccard

    def _confidence_score(self, candidate: str) -> float:
        tokens = candidate.lower().split()
        if not tokens:
            return 0.0

        # Penalise uncertainty words.
        uncertainty_count = sum(1 for t in tokens if t in _UNCERTAINTY_WORDS)
        uncertainty_penalty = min(uncertainty_count / len(tokens) * 5, 0.5)

        # Reward longer, more specific statements (up to a ceiling).
        length_bonus = min(len(tokens) / 30, 0.3)

        # Penalise very short fragments.
        brevity_penalty = 0.2 if len(tokens) < 4 else 0.0

        score = 0.7 + length_bonus - uncertainty_penalty - brevity_penalty
        return max(0.0, min(1.0, score))
