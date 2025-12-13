"""Document processors for different file formats."""

from processors.docx_processor import DOCXProcessor
from processors.html_processor import HTMLProcessor
from processors.image_processor import ImageProcessor
from processors.pdf_processor import PDFProcessor

__all__ = [
    "PDFProcessor",
    "DOCXProcessor",
    "HTMLProcessor",
    "ImageProcessor",
]
