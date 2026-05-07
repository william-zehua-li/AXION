import json
import math
import os
import re
from collections import defaultdict


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _TFIDFIndex:
    """Lightweight in-memory TF-IDF index — no external dependencies."""

    def __init__(self):
        self.chunks: list[str] = []
        # df[term] = number of documents containing term
        self._df: dict[str, int] = defaultdict(int)
        # tfidf[doc_idx] = {term: tf_idf_weight}
        self._tfidf: list[dict[str, float]] = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = list(chunks)
        self._df = defaultdict(int)
        self._tfidf = []

        term_sets = []
        for chunk in self.chunks:
            tokens = _tokenize(chunk)
            term_sets.append(tokens)
            for t in set(tokens):
                self._df[t] += 1

        n = len(self.chunks)
        for tokens in term_sets:
            tf: dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1.0
            for t in tf:
                tf[t] /= max(len(tokens), 1)
            vec = {
                t: tf[t] * math.log((n + 1) / (self._df[t] + 1))
                for t in tf
            }
            self._tfidf.append(vec)

    def add(self, chunk: str) -> None:
        """Incrementally add one chunk and update the index."""
        self.chunks.append(chunk)
        tokens = _tokenize(chunk)
        for t in set(tokens):
            self._df[t] += 1

        n = len(self.chunks)
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        for t in tf:
            tf[t] /= max(len(tokens), 1)
        new_vec = {
            t: tf[t] * math.log((n + 1) / (self._df[t] + 1))
            for t in tf
        }
        self._tfidf.append(new_vec)

        # IDF changed — cheaply rescale existing vecs for affected terms
        for i, vec in enumerate(self._tfidf[:-1]):
            for t in list(vec):
                if t in new_vec or t in set(tokens):
                    idf = math.log((n + 1) / (self._df[t] + 1))
                    old_idf = math.log(n / (self._df[t]))  # pre-add df
                    if old_idf > 0:
                        vec[t] = vec[t] / old_idf * idf

    def query(self, text: str, top_k: int) -> list[str]:
        if not self.chunks:
            return []

        tokens = _tokenize(text)
        n = len(self.chunks)
        q_vec: dict[str, float] = defaultdict(float)
        for t in tokens:
            q_vec[t] += 1.0
        for t in q_vec:
            q_vec[t] /= len(tokens)
            idf = math.log((n + 1) / (self._df.get(t, 0) + 1))
            q_vec[t] *= idf

        scores: list[tuple[float, int]] = []
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        for idx, vec in enumerate(self._tfidf):
            dot = sum(q_vec.get(t, 0.0) * w for t, w in vec.items())
            d_norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            scores.append((dot / (q_norm * d_norm), idx))

        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]


class FileStore:
    def __init__(self, store_path: str, queue_path: str):
        self._store_path = store_path
        self._queue_path = queue_path
        self._index = _TFIDFIndex()
        self._ensure_dirs()
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, chunk: str) -> None:
        """Write a new knowledge chunk to the store and update the index."""
        with open(self._store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": chunk}) + "\n")
        self._index.add(chunk)

    def queue(self, chunk: str) -> None:
        """Append a chunk to the pending-review queue (not in the main store)."""
        with open(self._queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": chunk}) + "\n")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top_k chunks most relevant to query."""
        return self._index.query(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        chunks: list[str] = []
        if os.path.exists(self._store_path):
            with open(self._store_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line)["text"])
                        except (json.JSONDecodeError, KeyError):
                            pass
        self._index.build(chunks)

    def _ensure_dirs(self) -> None:
        for path in (self._store_path, self._queue_path):
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
