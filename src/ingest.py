"""
Local Documentation Ingestion Engine.

Air-gapped ingestion of native host documentation (Linux man pages, local text/markdown):
- Decodes gzipped roff (.1.gz) manual pages from /usr/share/man
- Strips roff formatting macros (.TH, .SH, \fB, etc.) into clean lexical text
- Extracts local documentation files (.md, .txt) with length and token filters
- Guaranteed 100% offline with zero external network dependencies
"""

import glob
import gzip
import os
import re


def clean_roff_text(raw_roff: str) -> str:
    """
    Strips Unix roff/troff formatting directives to extract pure prose text.

    Removes:
    - Font escape sequences: \\fB, \\fI, \\fR, \\f(CW, etc.
    - Inline escape codes: \\-, \\e, \\(aq, etc.
    - Macro control lines: lines starting with '.', like .TH, .SH, .PP, .B, .I
    - Comments: .\\"
    """
    lines = []
    for line in raw_roff.splitlines():
        line = line.strip()
        # Skip macro directives and comments
        if not line or line.startswith((".", "'")):
            # Extract text from argument-bearing formatting lines (.B, .I, .BR, .IR)
            macro_match = re.match(r"^\.[A-Z]{1,2}\s+(.*)$", line)
            if macro_match:
                line = macro_match.group(1)
            else:
                continue

        # Strip font change escapes like \fB, \fI, \fR, \fP
        cleaned = re.sub(r"\\[fF]\[?[A-Za-z0-9]+\]?", "", line)
        # Strip character escapes like \-, \e, \(aq, \(em
        cleaned = re.sub(r"\\[-\w\(\)]+", " ", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 2:
            lines.append(cleaned)

    return " ".join(lines)


class LocalDocumentationIngester:
    """
    Extracts, cleans, and packages local host documentation for vectorization.
    """

    @staticmethod
    def load_man_pages(
        max_docs: int = 1000,
        man_dir: str = "/usr/share/man/man1",
        min_words: int = 20,
    ) -> list[tuple[str, str]]:
        """
        Ingests native Linux manual pages from a local man directory.

        Args:
            max_docs: Maximum number of man pages to load.
            man_dir: Path to directory containing man pages (e.g. /usr/share/man/man1).
            min_words: Minimum word threshold to exclude empty stub pages.

        Returns:
            List of (doc_id, cleaned_text) tuples.
        """
        if not os.path.exists(man_dir):
            raise FileNotFoundError(f"Man directory does not exist: {man_dir}")

        pattern = os.path.join(man_dir, "*.1.gz")
        all_paths = sorted(glob.glob(pattern))

        documents: list[tuple[str, str]] = []
        for path in all_paths:
            if len(documents) >= max_docs:
                break

            doc_id = os.path.basename(path).replace(".1.gz", "")
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    raw_content = f.read()
            except Exception:
                continue

            cleaned = clean_roff_text(raw_content)
            word_count = len(cleaned.split())
            if word_count >= min_words:
                documents.append((doc_id, cleaned))

        return documents

    @staticmethod
    def load_text_directory(
        directory: str,
        extensions: tuple[str, ...] = (".md", ".txt"),
        max_docs: int = 1000,
        min_words: int = 10,
    ) -> list[tuple[str, str]]:
        """
        Ingests local markdown or plain text files from a directory.

        Args:
            directory: Root path to search.
            extensions: Allowed file extensions.
            max_docs: Maximum number of files to ingest.
            min_words: Minimum word threshold.

        Returns:
            List of (doc_id, text_content) tuples.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents: list[tuple[str, str]] = []
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if len(documents) >= max_docs:
                    return documents

                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    rel_id = os.path.relpath(full_path, directory)
                    try:
                        with open(
                            full_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()
                    except Exception:
                        continue

                    # Collapse whitespace
                    cleaned = re.sub(r"\s+", " ", content).strip()
                    if len(cleaned.split()) >= min_words:
                        documents.append((rel_id, cleaned))

        return documents
