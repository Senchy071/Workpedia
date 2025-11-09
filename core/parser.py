"""Document parser using Docling for structure-aware extraction."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    VlmPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.vlm_pipeline import VlmPipeline

from config.config import PROJECT_ROOT, VLM_MODEL, VLM_PAGE_BATCH_SIZE

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Docling-based document parser with optimized settings.

    Features:
    - Multi-format support (PDF, DOCX, HTML, images)
    - DocLayNet + TableFormer models for layout analysis
    - GPU acceleration (when available)
    - Structure preservation (sections, tables, figures, cross-refs)
    - Memory-efficient processing
    """

    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        num_threads: int = 8,
        device: AcceleratorDevice = AcceleratorDevice.AUTO,
        backend: str = "v2",  # "v2", "pypdfium", or "vlm"
        vlm_model: str = VLM_MODEL,
    ):
        """
        Initialize document parser with optimized settings.

        Args:
            do_ocr: Enable OCR for scanned documents
            do_table_structure: Enable table structure recognition
            num_threads: Number of threads for processing
            device: Accelerator device (AUTO uses GPU if available)
            backend: PDF backend to use ("v2", "pypdfium", or "vlm")
            vlm_model: VLM model to use if backend="vlm" (e.g., "granite_docling")
        """
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.num_threads = num_threads
        self.device = device
        self.backend = backend
        self.vlm_model = vlm_model

        # Initialize converter (will be recreated as needed)
        self._converter: Optional[DocumentConverter] = None
        self._docs_processed = 0
        self._max_docs_before_recreation = 5  # Prevent memory leaks

        logger.info(
            f"DocumentParser initialized: OCR={do_ocr}, "
            f"TableStructure={do_table_structure}, Threads={num_threads}, "
            f"Device={device}, Backend={backend}"
            + (f", VLM_Model={vlm_model}" if backend == "vlm" else "")
        )

    def _get_converter(self) -> DocumentConverter:
        """
        Get or create DocumentConverter instance.

        Recreates converter every N documents to prevent memory leaks.
        """
        if (
            self._converter is None
            or self._docs_processed >= self._max_docs_before_recreation
        ):
            if self._converter is not None:
                logger.info(
                    f"Recreating converter after {self._docs_processed} documents"
                )
                # Allow garbage collection
                self._converter = None

            # Create converter based on backend type
            if self.backend == "vlm":
                # VLM Pipeline (Granite-Docling or SmolDocling)
                logger.info(f"Creating VLM pipeline with {self.vlm_model} model")

                # Configure global performance settings for batch processing
                settings.perf.page_batch_size = VLM_PAGE_BATCH_SIZE
                logger.info(f"Set page_batch_size to {VLM_PAGE_BATCH_SIZE} for parallel processing")

                # Use default Granite-Docling transformers configuration
                # The model selection is handled by the default VlmPipelineOptions
                vlm_options = VlmPipelineOptions()

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=vlm_options,
                        )
                    }
                )

                backend_name = f"VlmPipeline (granite-docling-258M with batch_size={VLM_PAGE_BATCH_SIZE})"

            else:
                # Standard PDF backends (v2 or pypdfium)
                if self.backend == "pypdfium":
                    backend_class = PyPdfiumDocumentBackend
                    backend_name = "PyPdfiumDocumentBackend"
                else:
                    backend_class = DoclingParseV2DocumentBackend
                    backend_name = "DoclingParseV2DocumentBackend"

                # Configure PDF pipeline options
                pipeline_options = PdfPipelineOptions(
                    do_ocr=self.do_ocr,
                    do_table_structure=self.do_table_structure,
                    accelerator_options=AcceleratorOptions(
                        num_threads=self.num_threads,
                        device=self.device,
                    ),
                )

                # Create converter with selected backend
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                            backend=backend_class,
                        )
                    }
                )

            self._docs_processed = 0
            logger.info(f"DocumentConverter created with {backend_name}")

        return self._converter

    def parse(
        self,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse document and extract structure.

        Args:
            file_path: Path to document file
            metadata: Optional additional metadata

        Returns:
            Dictionary containing:
                - doc_id: Unique document identifier
                - metadata: Document metadata
                - docling_document: Parsed DoclingDocument object
                - raw_text: Extracted text content
                - structure: Document structure info

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Get file stats
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        logger.info(f"Parsing document: {file_path.name} ({file_size_mb:.2f} MB)")
        start_time = datetime.now()

        try:
            # Get converter instance
            converter = self._get_converter()

            # Convert document
            result = converter.convert(str(file_path))

            # Extract DoclingDocument
            docling_doc = result.document

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract basic structure info
            num_pages = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0

            # Get text content
            raw_text = docling_doc.export_to_markdown()

            # Build metadata
            doc_metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size_mb": round(file_size_mb, 2),
                "pages": num_pages,
                "processing_time_seconds": round(processing_time, 2),
                "processed_in_chunks": False,
                "num_chunks": 1,
                "parsed_at": datetime.now().isoformat(),
                "parser_version": "docling_v2",
            }

            # Add custom metadata if provided
            if metadata:
                doc_metadata.update(metadata)

            # Increment counter
            self._docs_processed += 1

            logger.info(
                f"Document parsed successfully: {num_pages} pages in "
                f"{processing_time:.2f}s ({doc_metadata['filename']})"
            )

            return {
                "doc_id": doc_id,
                "metadata": doc_metadata,
                "docling_document": docling_doc,
                "raw_text": raw_text,
                "structure": {
                    "pages": num_pages,
                    "has_tables": self._has_tables(docling_doc),
                    "has_figures": self._has_figures(docling_doc),
                },
            }

        except Exception as e:
            logger.error(f"Failed to parse document {file_path.name}: {e}")
            raise

    def _has_tables(self, docling_doc) -> bool:
        """Check if document contains tables."""
        try:
            # Check for table elements in document
            for item in docling_doc.iterate_items():
                if hasattr(item, 'label') and 'table' in item.label.lower():
                    return True
            return False
        except Exception:
            return False

    def _has_figures(self, docling_doc) -> bool:
        """Check if document contains figures."""
        try:
            # Check for figure elements in document
            for item in docling_doc.iterate_items():
                if hasattr(item, 'label') and 'figure' in item.label.lower():
                    return True
            return False
        except Exception:
            return False

    def cleanup(self):
        """Clean up resources and free memory."""
        if self._converter is not None:
            logger.info("Cleaning up DocumentConverter")
            self._converter = None
            self._docs_processed = 0
