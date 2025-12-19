"""Query suggestion generator for Workpedia RAG system.

This module generates suggested queries from document structure:
- Section headings -> "What is X?" questions
- TOC entries -> "Tell me about [topic]" queries
- Key concepts extracted from content
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from config.config import (
    SUGGESTIONS_ENABLED,
    SUGGESTIONS_MAX_PER_DOCUMENT,
    SUGGESTIONS_MIN_HEADING_LENGTH,
    SUGGESTIONS_QUESTION_TEMPLATES,
)

logger = logging.getLogger(__name__)


@dataclass
class QuerySuggestion:
    """A suggested query for a document."""

    suggestion_id: str
    doc_id: str
    text: str
    source_type: str  # "heading", "toc", "concept", "topic"
    source_text: str  # Original text this was generated from
    priority: int = 1  # Higher = more relevant (1-10)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "source_type": self.source_type,
            "source_text": self.source_text,
            "priority": self.priority,
            "metadata": self.metadata,
        }


class QuerySuggestionGenerator:
    """
    Generate query suggestions from document structure.

    Extracts potential questions from:
    - Section headings ("## Introduction" -> "What is covered in the Introduction?")
    - TOC entries ("Chapter 1: Overview" -> "Tell me about Chapter 1: Overview")
    - Key concepts (noun phrases, technical terms)
    """

    def __init__(
        self,
        enabled: bool = SUGGESTIONS_ENABLED,
        max_suggestions: int = SUGGESTIONS_MAX_PER_DOCUMENT,
        min_heading_length: int = SUGGESTIONS_MIN_HEADING_LENGTH,
        question_templates: Optional[List[str]] = None,
    ):
        """
        Initialize suggestion generator.

        Args:
            enabled: Whether to generate suggestions
            max_suggestions: Maximum suggestions per document
            min_heading_length: Minimum heading length to process
            question_templates: Templates for generating questions
        """
        self.enabled = enabled
        self.max_suggestions = max_suggestions
        self.min_heading_length = min_heading_length
        self.question_templates = question_templates or SUGGESTIONS_QUESTION_TEMPLATES

        logger.info(
            f"QuerySuggestionGenerator initialized: enabled={enabled}, "
            f"max={max_suggestions}"
        )

    def generate_suggestions(
        self,
        parsed_doc: Dict[str, Any],
        include_headings: bool = True,
        include_toc: bool = True,
        include_concepts: bool = True,
    ) -> List[QuerySuggestion]:
        """
        Generate query suggestions from a parsed document.

        Args:
            parsed_doc: Output from DocumentParser.parse()
            include_headings: Generate from section headings
            include_toc: Generate from table of contents
            include_concepts: Generate from extracted concepts

        Returns:
            List of QuerySuggestion objects sorted by priority
        """
        if not self.enabled:
            return []

        doc_id = parsed_doc.get("doc_id", "unknown")
        suggestions: List[QuerySuggestion] = []
        seen_texts: Set[str] = set()  # Avoid duplicates

        logger.info(f"Generating suggestions for document: {doc_id}")

        # 1. Extract from section headings
        if include_headings:
            heading_suggestions = self._extract_from_headings(parsed_doc, doc_id, seen_texts)
            suggestions.extend(heading_suggestions)
            logger.debug(f"Generated {len(heading_suggestions)} heading suggestions")

        # 2. Extract from TOC (if available in structure)
        if include_toc:
            toc_suggestions = self._extract_from_toc(parsed_doc, doc_id, seen_texts)
            suggestions.extend(toc_suggestions)
            logger.debug(f"Generated {len(toc_suggestions)} TOC suggestions")

        # 3. Extract key concepts (from metadata or content analysis)
        if include_concepts:
            concept_suggestions = self._extract_concepts(parsed_doc, doc_id, seen_texts)
            suggestions.extend(concept_suggestions)
            logger.debug(f"Generated {len(concept_suggestions)} concept suggestions")

        # Sort by priority (highest first) and limit
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        suggestions = suggestions[: self.max_suggestions]

        logger.info(f"Generated {len(suggestions)} total suggestions for {doc_id}")
        return suggestions

    def _extract_from_headings(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
        seen_texts: Set[str],
    ) -> List[QuerySuggestion]:
        """Extract suggestions from markdown headings in document text."""
        suggestions = []
        raw_text = parsed_doc.get("raw_text", "")

        if not raw_text:
            return suggestions

        # Find markdown headings (## Heading)
        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        matches = heading_pattern.findall(raw_text)

        for level_markers, heading_text in matches:
            heading_text = heading_text.strip()
            level = len(level_markers)

            # Skip short or generic headings
            if len(heading_text) < self.min_heading_length:
                continue
            if self._is_generic_heading(heading_text):
                continue

            # Clean the heading text
            heading_text = self._clean_heading(heading_text)
            if not heading_text or heading_text.lower() in seen_texts:
                continue

            seen_texts.add(heading_text.lower())

            # Generate question from heading
            question = self._heading_to_question(heading_text, level)
            if question:
                # Priority based on heading level (h1=10, h2=8, h3=6, h4=4)
                priority = max(1, 12 - level * 2)

                suggestions.append(
                    QuerySuggestion(
                        suggestion_id=f"{doc_id}_heading_{len(suggestions)}",
                        doc_id=doc_id,
                        text=question,
                        source_type="heading",
                        source_text=heading_text,
                        priority=priority,
                        metadata={"heading_level": level},
                    )
                )

        return suggestions

    def _extract_from_toc(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
        seen_texts: Set[str],
    ) -> List[QuerySuggestion]:
        """Extract suggestions from document structure/TOC."""
        suggestions = []

        # Try to get sections from docling_document structure
        docling_doc = parsed_doc.get("docling_document")
        if docling_doc:
            try:
                from core.analyzer import StructureAnalyzer

                analyzer = StructureAnalyzer()
                structure = analyzer.analyze(docling_doc)

                for section in structure.sections:
                    section_text = section.get("text", "").strip()
                    section_level = section.get("level", 1)

                    if len(section_text) < self.min_heading_length:
                        continue
                    if self._is_generic_heading(section_text):
                        continue

                    section_text = self._clean_heading(section_text)
                    if not section_text or section_text.lower() in seen_texts:
                        continue

                    seen_texts.add(section_text.lower())

                    # Generate "Tell me about X" style questions
                    question = f"Tell me about {section_text}"
                    priority = max(1, 10 - section_level)

                    suggestions.append(
                        QuerySuggestion(
                            suggestion_id=f"{doc_id}_toc_{len(suggestions)}",
                            doc_id=doc_id,
                            text=question,
                            source_type="toc",
                            source_text=section_text,
                            priority=priority,
                            metadata={
                                "section_level": section_level,
                                "page": section.get("page"),
                            },
                        )
                    )

            except Exception as e:
                logger.debug(f"Could not extract TOC structure: {e}")

        return suggestions

    def _extract_concepts(
        self,
        parsed_doc: Dict[str, Any],
        doc_id: str,
        seen_texts: Set[str],
    ) -> List[QuerySuggestion]:
        """Extract key concepts from document content."""
        suggestions = []
        raw_text = parsed_doc.get("raw_text", "")

        if not raw_text:
            return suggestions

        # Extract capitalized multi-word phrases (likely key concepts)
        # Pattern: Two or more capitalized words in sequence
        concept_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b"
        )
        concepts = concept_pattern.findall(raw_text)

        # Count occurrences to find important concepts
        concept_counts: Dict[str, int] = {}
        for concept in concepts:
            concept = concept.strip()
            if len(concept) >= self.min_heading_length:
                concept_lower = concept.lower()
                concept_counts[concept_lower] = concept_counts.get(concept_lower, 0) + 1

        # Sort by frequency and take top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)

        for concept_lower, count in sorted_concepts[:15]:  # Top 15 concepts
            if concept_lower in seen_texts:
                continue
            if self._is_generic_heading(concept_lower):
                continue

            seen_texts.add(concept_lower)

            # Find the properly capitalized version
            for concept in concepts:
                if concept.lower() == concept_lower:
                    # Generate question
                    question = f"What is {concept}?"
                    priority = min(7, 3 + count)  # Higher frequency = higher priority (max 7)

                    suggestions.append(
                        QuerySuggestion(
                            suggestion_id=f"{doc_id}_concept_{len(suggestions)}",
                            doc_id=doc_id,
                            text=question,
                            source_type="concept",
                            source_text=concept,
                            priority=priority,
                            metadata={"frequency": count},
                        )
                    )
                    break

        return suggestions

    def _heading_to_question(self, heading: str, level: int) -> Optional[str]:
        """
        Convert a heading to a question.

        Args:
            heading: The heading text
            level: Heading level (1-4)

        Returns:
            Generated question or None
        """
        heading = heading.strip()
        if not heading:
            return None

        # Choose template based on heading characteristics
        heading_lower = heading.lower()

        # If heading is already a question, use it
        if heading.endswith("?"):
            return heading

        # Handle numbered chapters/sections
        if re.match(r"^(?:chapter|section|part)\s+\d", heading_lower):
            return f"What is covered in {heading}?"

        # Handle "Introduction to X" patterns
        if heading_lower.startswith("introduction"):
            return f"What does the {heading} cover?"

        # Handle "X Overview" patterns
        if heading_lower.endswith("overview"):
            return f"Can you provide an {heading}?"

        # Handle "How to X" patterns (already question-like)
        if heading_lower.startswith("how to"):
            return f"{heading}?"

        # Handle definitions (starts with "What is", etc.)
        if heading_lower.startswith(("what ", "why ", "when ", "where ", "who ")):
            return f"{heading}?"

        # For h1/h2 level headings, use broader questions
        if level <= 2:
            return f"What is covered in the {heading} section?"

        # Default: "What is X?" style question
        return f"What is {heading}?"

    def _clean_heading(self, heading: str) -> str:
        """Clean heading text for use in questions."""
        # Remove leading/trailing whitespace and special chars
        heading = heading.strip()

        # Remove markdown formatting
        heading = re.sub(r"\*\*(.+?)\*\*", r"\1", heading)  # Bold
        heading = re.sub(r"\*(.+?)\*", r"\1", heading)  # Italic
        heading = re.sub(r"`(.+?)`", r"\1", heading)  # Code
        heading = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", heading)  # Links

        # Remove excessive whitespace
        heading = re.sub(r"\s+", " ", heading)

        # Remove trailing punctuation (except ?)
        heading = re.sub(r"[.:;,]+$", "", heading)

        return heading.strip()

    def _is_generic_heading(self, heading: str) -> bool:
        """Check if heading is too generic to make a good question."""
        generic_headings = {
            "introduction",
            "conclusion",
            "summary",
            "overview",
            "background",
            "references",
            "bibliography",
            "appendix",
            "appendices",
            "index",
            "table of contents",
            "contents",
            "acknowledgments",
            "acknowledgements",
            "preface",
            "foreword",
            "glossary",
            "abbreviations",
            "acronyms",
            "list of figures",
            "list of tables",
            "notes",
            "abstract",
        }

        heading_lower = heading.lower().strip()

        # Check exact match
        if heading_lower in generic_headings:
            return True

        # Check if mostly numbers/symbols
        alpha_chars = sum(1 for c in heading if c.isalpha())
        if alpha_chars < len(heading) * 0.5:
            return True

        return False

    def get_default_suggestions(self, doc_id: str, filename: str) -> List[QuerySuggestion]:
        """
        Generate default suggestions when document-specific ones aren't available.

        Args:
            doc_id: Document ID
            filename: Document filename

        Returns:
            List of generic suggestions for this document
        """
        base_name = re.sub(r"\.[^.]+$", "", filename)  # Remove extension
        base_name = re.sub(r"[-_]", " ", base_name)  # Replace separators with spaces

        suggestions = [
            QuerySuggestion(
                suggestion_id=f"{doc_id}_default_0",
                doc_id=doc_id,
                text="What is this document about?",
                source_type="default",
                source_text=filename,
                priority=10,
            ),
            QuerySuggestion(
                suggestion_id=f"{doc_id}_default_1",
                doc_id=doc_id,
                text=f"What are the main topics in {base_name}?",
                source_type="default",
                source_text=filename,
                priority=9,
            ),
            QuerySuggestion(
                suggestion_id=f"{doc_id}_default_2",
                doc_id=doc_id,
                text="Summarize the key points",
                source_type="default",
                source_text=filename,
                priority=8,
            ),
            QuerySuggestion(
                suggestion_id=f"{doc_id}_default_3",
                doc_id=doc_id,
                text="What are the conclusions?",
                source_type="default",
                source_text=filename,
                priority=7,
            ),
        ]

        return suggestions[: self.max_suggestions]


def generate_suggestions(
    parsed_doc: Dict[str, Any],
    max_suggestions: int = SUGGESTIONS_MAX_PER_DOCUMENT,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate suggestions from a parsed document.

    Args:
        parsed_doc: Output from DocumentParser.parse()
        max_suggestions: Maximum suggestions to generate

    Returns:
        List of suggestion dictionaries
    """
    generator = QuerySuggestionGenerator(max_suggestions=max_suggestions)
    suggestions = generator.generate_suggestions(parsed_doc)
    return [s.to_dict() for s in suggestions]
