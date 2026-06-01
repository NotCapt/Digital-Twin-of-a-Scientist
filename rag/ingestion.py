"""
RAG Pipeline — Document Ingestion Module

Handles loading and preprocessing of Alan Turing's source materials
from various formats (PDF, TXT, Markdown).
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


class DocumentIngestion:
    """Loads and preprocesses documents from the data/raw directory."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}

    def __init__(self, raw_data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_all_documents(self) -> list[dict]:
        """
        Load all documents from the raw data directory.

        Returns:
            List of document dicts with keys:
            - text: str — the full extracted text
            - metadata: dict — source_title, source_type, year, file_path, is_turing_voice
        """
        documents = []

        if not self.raw_data_dir.exists():
            print(f"[Ingestion] Warning: Raw data directory '{self.raw_data_dir}' does not exist.")
            return documents

        for category_dir in self.raw_data_dir.iterdir():
            if not category_dir.is_dir():
                continue

            source_type = category_dir.name  # papers, books, interviews, letters

            for file_path in category_dir.rglob("*"):
                if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                doc = self._load_single_document(file_path, source_type)
                if doc and doc["text"].strip():
                    documents.append(doc)
                    print(f"[Ingestion] Loaded: {file_path.name} ({len(doc['text'])} chars)")

        print(f"[Ingestion] Total documents loaded: {len(documents)}")
        return documents

    def _load_single_document(self, file_path: Path, source_type: str) -> Optional[dict]:
        """Load a single document and extract text + metadata."""
        try:
            text = self._extract_text(file_path)
            if not text:
                return None

            # Clean the text
            text = self._clean_text(text)

            # Extract metadata from filename convention: "YEAR - Title.ext"
            metadata = self._extract_metadata(file_path, source_type)

            return {
                "text": text,
                "metadata": metadata,
                "doc_id": self._generate_doc_id(file_path),
            }
        except Exception as e:
            print(f"[Ingestion] Error loading {file_path}: {e}")
            return None

    def _extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a file based on its extension."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf_text(file_path)
        elif ext in {".txt", ".md", ".text"}:
            return self._extract_plain_text(file_path)
        return None

    def _extract_pdf_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF file using pdfplumber or PyPDF2."""
        # Try pdfplumber first (better quality)
        if pdfplumber:
            try:
                with pdfplumber.open(file_path) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                    return "\n\n".join(pages)
            except Exception:
                pass

        # Fallback to PyPDF2
        if PdfReader:
            try:
                reader = PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except Exception:
                pass

        print(f"[Ingestion] No PDF reader available for {file_path}")
        return None

    def _extract_plain_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a plain text or markdown file."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove artifacts."""
        import re

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove common PDF artifacts
        text = re.sub(r"\x0c", "\n", text)  # Form feed characters

        # Normalize spaces (but not newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Clean up lines
        lines = text.split("\n")
        cleaned_lines = [line.strip() for line in lines]
        text = "\n".join(cleaned_lines)

        return text.strip()

    def _extract_metadata(self, file_path: Path, source_type: str) -> dict:
        """
        Extract metadata from the file.
        Filename convention: "YEAR - Title.ext" or just "Title.ext"
        """
        import re

        filename = file_path.stem
        year = None
        title = filename

        # Try to extract year from filename: "1936 - On Computable Numbers"
        year_match = re.match(r"^(\d{4})\s*[-–—]\s*(.+)$", filename)
        if year_match:
            year = int(year_match.group(1))
            title = year_match.group(2).strip()

        # Determine if this is Turing's own voice or a secondary source
        is_turing_voice = source_type in {"papers", "letters", "lectures"}

        # Check for sidecar metadata file
        meta_path = file_path.with_suffix(".meta.json")
        extra_meta = {}
        if meta_path.exists():
            try:
                extra_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        metadata = {
            "source_title": extra_meta.get("title", title),
            "source_type": source_type,
            "year": extra_meta.get("year", year),
            "author": extra_meta.get("author", "Alan Turing" if is_turing_voice else "Unknown"),
            "is_turing_voice": extra_meta.get("is_turing_voice", is_turing_voice),
            "file_path": str(file_path),
        }

        return metadata

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a stable document ID from the file path."""
        return hashlib.md5(str(file_path).encode()).hexdigest()

    def save_processed(self, documents: list[dict], output_file: str = "documents.json"):
        """Save processed documents to a JSON file for inspection."""
        output_path = self.processed_dir / output_file
        serializable = []
        for doc in documents:
            serializable.append({
                "doc_id": doc["doc_id"],
                "metadata": doc["metadata"],
                "text_length": len(doc["text"]),
                "text_preview": doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
            })

        output_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Ingestion] Saved processed documents manifest to {output_path}")
