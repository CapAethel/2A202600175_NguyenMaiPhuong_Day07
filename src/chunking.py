from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])(?:\s|\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            # No separators left — force split by chunk_size
            chunks: list[str] = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i:i + self.chunk_size])
            return chunks
        sep = remaining_separators[0]
        if sep == "":
            return self._split(current_text, remaining_separators[1:])
        parts = current_text.split(sep)
        if len(parts) <= 1:
            return self._split(current_text, remaining_separators[1:])
        result: list[str] = []
        current_chunk = parts[0]
        for part in parts[1:]:
            candidate = current_chunk + sep + part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    result.extend(self._split(current_chunk, remaining_separators[1:]))
                current_chunk = part
        if current_chunk:
            result.extend(self._split(current_chunk, remaining_separators[1:]))
        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            'fixed_size': FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text),
            'by_sentences': SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            'recursive': RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            result[name] = {
                'count': count,
                'avg_length': avg_length,
                'chunks': chunks,
            }
        return result


class ParentChildChunker:
    """
    Parent-Child (Small-to-Big) chunking strategy.

    1. Split document into **parent** chunks by section headings
       (chapters, roman-numeral sections, numbered subsections).
    2. Each parent is further split into **child** chunks using
       SentenceChunker for retrieval-friendly granularity.
    3. Every child carries a ``parent_id`` so the retrieval layer can
       "expand" a matched child back to its full parent context.

    Designed for Vietnamese academic textbooks with bold markdown headings
    like ``**CHƯƠNG I**``, ``**I. …**``, ``**1\\. …**``.
    """

    # Regex that matches the heading patterns found in the textbook:
    #   CHƯƠNG I / CHƯƠNG 2 / CHƯƠNG III  (chapter)
    #   **I. TITLE**                      (section)
    #   **1\. Title**                     (subsection)
    #   ***a. Title***                    (sub-subsection)
    _HEADING_RE = re.compile(
        r"^(?:"
        r"CHƯƠNG\s+[IVXLCDM\d]+"                      # bare chapter line
        r"|"
        r"\*{2,3}(?:CHƯƠNG\s+[IVXLCDM\d]+.*?)\*{2,3}" # bold chapter
        r"|"
        r"\*{2,3}[IVX]+\\\.\s.+?\*{2,3}"               # bold roman section
        r"|"
        r"\*{2,3}\d+\\\.\s.+?\*{2,3}"                   # bold numbered subsection
        r"|"
        r"\*{2,3}[a-e]\.\s.+?\*{2,3}"                   # italic/bold lettered sub-subsection
        r")",
        re.MULTILINE,
    )

    def __init__(
        self,
        child_sentences: int = 3,
        child_max_chars: int = 500,
    ) -> None:
        self.child_sentences = child_sentences
        self.child_max_chars = child_max_chars

    # ── public API ──────────────────────────────────────────────

    def chunk(self, text: str) -> list[dict]:
        """
        Return a flat list of child-chunk dicts, each containing:
            - ``child_id``:   ``"p{parent_idx}_c{child_idx}"``
            - ``parent_id``:  ``"p{parent_idx}"``
            - ``content``:    the child text (small chunk for embedding)
            - ``parent_content``: the full parent text (big chunk for LLM context)
            - ``heading``:    the section heading that starts this parent
        """
        if not text:
            return []

        parents = self._split_parents(text)
        child_chunker = SentenceChunker(max_sentences_per_chunk=self.child_sentences)

        all_children: list[dict] = []
        for p_idx, (heading, parent_text) in enumerate(parents):
            children = child_chunker.chunk(parent_text)
            # If any child exceeds max chars, re-split with FixedSizeChunker
            refined: list[str] = []
            for c in children:
                if len(c) > self.child_max_chars:
                    refined.extend(
                        FixedSizeChunker(
                            chunk_size=self.child_max_chars, overlap=50
                        ).chunk(c)
                    )
                else:
                    refined.append(c)

            for c_idx, child_text in enumerate(refined):
                all_children.append(
                    {
                        "child_id": f"p{p_idx}_c{c_idx}",
                        "parent_id": f"p{p_idx}",
                        "content": child_text,
                        "parent_content": parent_text,
                        "heading": heading,
                    }
                )
        return all_children

    # ── internals ───────────────────────────────────────────────

    def _split_parents(self, text: str) -> list[tuple[str, str]]:
        """Split text into (heading, body) pairs using heading regex."""
        matches = list(self._HEADING_RE.finditer(text))
        if not matches:
            return [("(no heading)", text)]

        parents: list[tuple[str, str]] = []

        # Text before the first heading (preamble)
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                parents.append(("(preamble)", preamble))

        for i, m in enumerate(matches):
            heading = m.group(0).strip().strip("*")
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                parents.append((heading, body))

        return parents
