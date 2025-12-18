"""Document summarization using Ollama LLM.

This module generates executive summaries for indexed documents,
providing users with a quick overview before querying.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.config import (
    SUMMARY_ENABLED,
    SUMMARY_MAX_BULLETS,
    SUMMARY_MAX_INPUT_CHARS,
    SUMMARY_TEMPERATURE,
)
from core.llm import OllamaClient

logger = logging.getLogger(__name__)


# System prompt for document summarization
SUMMARY_SYSTEM_PROMPT = """You are a document summarization assistant. Your task is to create
concise, informative executive summaries of documents.

Guidelines:
- Create exactly {num_bullets} bullet points summarizing the key content
- Focus on the main topics, findings, and important information
- Use clear, professional language
- Each bullet should be a complete, standalone point
- Prioritize the most important information first
- Do not include meta-commentary like "This document discusses..."
- Start each bullet with a dash (-) followed by a space

Output format:
- First key point about the document
- Second key point about the document
- (continue for all bullets)
"""


@dataclass
class DocumentSummary:
    """
    Document summary result.

    Attributes:
        doc_id: Document identifier
        summary: Generated summary text
        bullets: List of individual bullet points
        metadata: Additional summary metadata
    """

    doc_id: str
    summary: str
    bullets: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "summary": self.summary,
            "bullets": self.bullets,
            "metadata": self.metadata,
        }


class DocumentSummarizer:
    """
    Generates executive summaries for documents using Ollama LLM.

    The summarizer extracts key information from document content and
    creates a concise bullet-point summary that helps users understand
    what the document contains before querying.

    Usage:
        summarizer = DocumentSummarizer()
        summary = summarizer.summarize(parsed_doc)
        print(summary.bullets)
    """

    def __init__(
        self,
        llm: Optional[OllamaClient] = None,
        max_bullets: int = SUMMARY_MAX_BULLETS,
        max_input_chars: int = SUMMARY_MAX_INPUT_CHARS,
        temperature: float = SUMMARY_TEMPERATURE,
        enabled: bool = SUMMARY_ENABLED,
    ):
        """
        Initialize document summarizer.

        Args:
            llm: OllamaClient instance (creates default if None)
            max_bullets: Maximum number of summary bullet points (3-7)
            max_input_chars: Maximum characters to send to LLM for summarization
            temperature: LLM temperature for generation (lower = more focused)
            enabled: Whether summarization is enabled
        """
        self.llm = llm or OllamaClient()
        self.max_bullets = max(3, min(7, max_bullets))  # Clamp between 3-7
        self.max_input_chars = max_input_chars
        self.temperature = temperature
        self.enabled = enabled

        logger.info(
            f"DocumentSummarizer initialized: enabled={enabled}, "
            f"max_bullets={self.max_bullets}, max_input_chars={max_input_chars}"
        )

    def summarize(
        self,
        parsed_doc: Dict[str, Any],
        force: bool = False,
    ) -> Optional[DocumentSummary]:
        """
        Generate a summary for a parsed document.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            force: Generate summary even if disabled

        Returns:
            DocumentSummary object, or None if summarization is disabled/fails
        """
        if not self.enabled and not force:
            logger.debug("Summarization is disabled")
            return None

        doc_id = parsed_doc.get("doc_id", "unknown")
        filename = parsed_doc.get("metadata", {}).get("filename", "unknown")

        logger.info(f"Generating summary for document: {filename} ({doc_id})")

        # Extract content for summarization
        content = self._extract_content(parsed_doc)

        if not content or len(content.strip()) < 100:
            logger.warning(f"Insufficient content for summarization: {doc_id}")
            return None

        # Generate summary using LLM
        try:
            summary_text = self._generate_summary(content, filename)
            bullets = self._parse_bullets(summary_text)

            if not bullets:
                logger.warning(f"Failed to parse bullets from summary: {doc_id}")
                return None

            summary = DocumentSummary(
                doc_id=doc_id,
                summary=summary_text,
                bullets=bullets,
                metadata={
                    "filename": filename,
                    "num_bullets": len(bullets),
                    "input_chars": len(content),
                    "model": self.llm.model,
                },
            )

            logger.info(
                f"Generated summary for {filename}: {len(bullets)} bullets, "
                f"{len(summary_text)} chars"
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {doc_id}: {e}")
            return None

    def _extract_content(self, parsed_doc: Dict[str, Any]) -> str:
        """
        Extract content from parsed document for summarization.

        Prioritizes structured content (sections, headings) over raw text,
        and truncates to max_input_chars.
        """
        content_parts = []

        # Try to get document title/filename
        metadata = parsed_doc.get("metadata", {})
        filename = metadata.get("filename", "")
        if filename:
            content_parts.append(f"Document: {filename}\n")

        # Add page count if available
        pages = metadata.get("pages", 0)
        if pages:
            content_parts.append(f"Pages: {pages}\n\n")

        # Get raw text content
        raw_text = parsed_doc.get("raw_text", "")

        # Try to get structured content from TOC or sections
        docling_doc = parsed_doc.get("docling_document")
        if docling_doc:
            try:
                from core.analyzer import StructureAnalyzer

                analyzer = StructureAnalyzer()
                structure = analyzer.analyze(docling_doc)

                # Add section titles for context
                if structure.sections:
                    section_titles = [
                        s.get("text", "") for s in structure.sections[:20]
                        if s.get("text")
                    ]
                    if section_titles:
                        content_parts.append("Main sections:\n")
                        for title in section_titles:
                            content_parts.append(f"- {title}\n")
                        content_parts.append("\n")

            except Exception as e:
                logger.debug(f"Could not extract structure: {e}")

        # Add raw text content
        if raw_text:
            content_parts.append("Content:\n")
            content_parts.append(raw_text)

        # Combine and truncate
        full_content = "".join(content_parts)

        if len(full_content) > self.max_input_chars:
            # Truncate intelligently at sentence boundary if possible
            truncated = full_content[: self.max_input_chars]
            last_period = truncated.rfind(".")
            if last_period > self.max_input_chars * 0.8:
                truncated = truncated[: last_period + 1]
            full_content = truncated + "\n\n[Content truncated for summarization]"

        return full_content

    def _generate_summary(self, content: str, filename: str) -> str:
        """Generate summary using LLM."""
        system_prompt = SUMMARY_SYSTEM_PROMPT.format(num_bullets=self.max_bullets)

        user_prompt = f"""Please summarize the following document in exactly \
{self.max_bullets} bullet points.
Focus on the main topics, key findings, and important information.

{content}

Provide exactly {self.max_bullets} bullet points, each starting with a dash (-):"""

        response = self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
            stream=False,
        )

        return response.strip()

    def _parse_bullets(self, summary_text: str) -> List[str]:
        """Parse bullet points from summary text."""
        bullets = []
        lines = summary_text.split("\n")

        for line in lines:
            line = line.strip()
            # Handle various bullet formats: -, *, •, numbered
            if line.startswith(("-", "*", "•")):
                bullet = line[1:].strip()
                if bullet:
                    bullets.append(bullet)
            elif line and line[0].isdigit() and ("." in line[:3] or ")" in line[:3]):
                # Numbered list: "1. text" or "1) text"
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                if len(parts) > 1:
                    bullet = parts[1].strip()
                    if bullet:
                        bullets.append(bullet)

        return bullets

    def format_summary_for_chunk(self, summary: DocumentSummary) -> str:
        """
        Format summary as content suitable for storage as a chunk.

        This creates a searchable text representation of the summary
        that can be retrieved for "what's in this document" queries.
        """
        lines = [
            "# DOCUMENT SUMMARY",
            "",
            "This is the executive summary of this document.",
            "This summary provides an overview of the document's main content and key points.",
            "Document overview and summary of main topics:",
            "",
            f"Document: {summary.metadata.get('filename', 'Unknown')}",
            "",
            "Key Points:",
            "",
        ]

        for i, bullet in enumerate(summary.bullets, 1):
            lines.append(f"{i}. {bullet}")

        lines.append("")
        lines.append("This summary was automatically generated to help users understand")
        lines.append("what information is contained in this document.")

        return "\n".join(lines)

    def is_summary_query(self, query: str) -> bool:
        """
        Detect if a query is asking for a document summary/overview.

        Args:
            query: User's query text

        Returns:
            True if query is asking about document content/summary
        """
        query_lower = query.lower()

        summary_keywords = [
            "what is in this document",
            "what's in this document",
            "what does this document",
            "document summary",
            "document overview",
            "summarize this document",
            "summarize the document",
            "summary of this document",
            "overview of this document",
            "what is this document about",
            "what's this document about",
            "main topics",
            "main points",
            "key points",
            "key topics",
            "document about",
            "tell me about this document",
            "describe this document",
            "what does it cover",
            "what does it contain",
        ]

        return any(kw in query_lower for kw in summary_keywords)
