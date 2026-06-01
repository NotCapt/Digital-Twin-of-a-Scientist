"""
RAG Pipeline — Text Chunking Module

Splits documents into overlapping chunks with rich metadata,
optimized for semantic retrieval over Turing's works.
"""

import re
from typing import Optional


class TextChunker:
    """Splits documents into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target number of tokens per chunk (approximated as words).
            chunk_overlap: Number of tokens to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Split a document into chunks with metadata.

        Args:
            document: Dict with 'text', 'metadata', and 'doc_id'.

        Returns:
            List of chunk dicts with 'text', 'metadata', 'chunk_id'.
        """
        text = document["text"]
        metadata = document["metadata"]
        doc_id = document["doc_id"]

        # First, try to split by sections/headings
        sections = self._split_into_sections(text)

        chunks = []
        global_chunk_index = 0

        for section_title, section_text in sections:
            # Split each section into token-sized chunks with overlap
            section_chunks = self._split_with_overlap(section_text)

            for i, chunk_text in enumerate(section_chunks):
                if not chunk_text.strip():
                    continue

                chunk = {
                    "text": chunk_text.strip(),
                    "chunk_id": f"{doc_id}_chunk_{global_chunk_index}",
                    "metadata": {
                        **metadata,
                        "section": section_title,
                        "chunk_index": global_chunk_index,
                        "section_chunk_index": i,
                    },
                }
                chunks.append(chunk)
                global_chunk_index += 1

        # Add total_chunks to each chunk's metadata
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        return chunks

    def chunk_all_documents(self, documents: list[dict]) -> list[dict]:
        """Chunk all documents and return a flat list of chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(
                f"[Chunker] '{doc['metadata'].get('source_title', 'Unknown')}' "
                f"-> {len(chunks)} chunks"
            )

        print(f"[Chunker] Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into sections based on headings or structure.

        Returns:
            List of (section_title, section_text) tuples.
        """
        # Common section patterns in academic papers and documents
        section_patterns = [
            r"^(?:#{1,3})\s+(.+)$",                    # Markdown headings
            r"^(?:SECTION|Section|CHAPTER|Chapter)\s+\d+[.:]\s*(.+)$",
            r"^(\d+\.(?:\d+\.)*)\s+(.+)$",             # Numbered sections like "1. Introduction"
            r"^([A-Z][A-Z\s]{3,})$",                    # ALL CAPS headings
        ]

        combined_pattern = "|".join(f"(?:{p})" for p in section_patterns)

        sections = []
        current_title = "Introduction"
        current_lines = []

        for line in text.split("\n"):
            is_heading = False

            # Check against each pattern
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save the current section
                    if current_lines:
                        sections.append((current_title, "\n".join(current_lines)))

                    # Extract the new section title
                    groups = [g for g in match.groups() if g]
                    current_title = groups[-1].strip() if groups else line.strip()
                    current_lines = []
                    is_heading = True
                    break

            if not is_heading:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        # If no sections were found, treat the entire text as one section
        if not sections:
            sections = [("Full Document", text)]

        return sections

    def _split_with_overlap(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks by word count.
        Uses sentence boundaries when possible for cleaner splits.
        """
        # Split into sentences first for cleaner boundaries
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk_words = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            # If a single sentence exceeds chunk size, force-split it
            if sentence_word_count > self.chunk_size:
                # Save current chunk if non-empty
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))

                # Force-split the long sentence
                for i in range(0, sentence_word_count, self.chunk_size - self.chunk_overlap):
                    chunk_words = sentence_words[i : i + self.chunk_size]
                    if chunk_words:
                        chunks.append(" ".join(chunk_words))

                current_chunk_words = []
                current_word_count = 0
                continue

            # Check if adding this sentence exceeds chunk size
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk_words:
                # Save current chunk
                chunks.append(" ".join(current_chunk_words))

                # Start new chunk with overlap from the end of the previous chunk
                overlap_words = current_chunk_words[-self.chunk_overlap :] if self.chunk_overlap > 0 else []
                current_chunk_words = overlap_words + sentence_words
                current_word_count = len(current_chunk_words)
            else:
                current_chunk_words.extend(sentence_words)
                current_word_count += sentence_word_count

        # Don't forget the last chunk
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex-based heuristics."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r"(\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g))\.", r"\1<DOT>", text)

        # Split on sentence-ending punctuation followed by space and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Restore abbreviation dots
        sentences = [s.replace("<DOT>", ".") for s in sentences]

        # Also split on double newlines (paragraph breaks)
        final_sentences = []
        for sentence in sentences:
            parts = re.split(r"\n\s*\n", sentence)
            final_sentences.extend(parts)

        # Filter empty sentences
        return [s.strip() for s in final_sentences if s.strip()]
