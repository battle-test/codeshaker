from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List


TEXT_EXTENSIONS = {
    ".go",
    ".ts",
    ".tsx",
    ".js",
    ".py",
    ".java",
    ".kt",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".json",
    ".proto",
    ".sql",
    ".txt",
}


@dataclass
class Chunk:
    chunk_id: int
    start: int
    end: int
    content: str
    preview: str

    def doc_id(self, repo: str, branch: str, path: str, file_sha: str) -> str:
        raw = f"{repo}|{branch}|{path}|{self.chunk_id}|{file_sha}".encode()
        return hashlib.sha256(raw).hexdigest()


def is_textual(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in TEXT_EXTENSIONS


def iter_chunks(text: str, max_chars: int, overlap: int) -> Iterator[Chunk]:
    if not text:
        return iter([])

    start = 0
    chunk_id = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chars, text_length)
        content = text[start:end]
        preview = content[:240].replace("\n", " ")
        yield Chunk(chunk_id=chunk_id, start=start, end=end, content=content, preview=preview)
        chunk_id += 1
        if end == text_length:
            break
        start = max(0, end - overlap)


def batched(iterable: Iterable[Chunk], size: int) -> Iterator[List[Chunk]]:
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch
