#!/usr/bin/env python3
"""Demo script for testing document parser functionality."""

import logging
from pathlib import Path

from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.html_processor import HTMLProcessor
from processors.image_processor import ImageProcessor
from core.validator import DocumentValidator
from core.progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_processor(file_path: Path):
    """Select appropriate processor based on file type."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PDFProcessor()
    elif suffix in [".docx", ".doc"]:
        return DOCXProcessor()
    elif suffix in [".html", ".htm"]:
        return HTMLProcessor()
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        return ImageProcessor()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def process_document(file_path: Path):
    """
    Process a single document and validate results.

    Args:
        file_path: Path to document file
    """
    print(f"\n{'='*70}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*70}\n")

    try:
        # Select processor
        processor = get_processor(file_path)
        print(f"Using processor: {processor.__class__.__name__}")

        # Process document
        result = processor.process(
            file_path=file_path,
            analyze_structure=True,
        )

        # Display results
        print(f"\nDocument ID: {result['doc_id']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")

        print(f"\nStructure:")
        for key, value in result['structure'].items():
            print(f"  {key}: {value}")

        if 'structure_analysis' in result:
            print(f"\nStructure Analysis:")
            analysis = result['structure_analysis']
            if 'sections' in analysis:
                print(f"  Sections: {len(analysis['sections'])}")
            if 'tables' in analysis:
                print(f"  Tables: {len(analysis['tables'])}")
            if 'figures' in analysis:
                print(f"  Figures: {len(analysis['figures'])}")

        # Validate results
        print(f"\nValidating results...")
        validator = DocumentValidator()
        validation_report = validator.validate(result)

        print(f"\nValidation Status: {'✓ VALID' if validation_report.is_valid else '✗ INVALID'}")
        print(f"Checks Passed: {validation_report.passed_checks}/{validation_report.total_checks}")
        print(f"Pass Rate: {validation_report.summary['pass_rate']}%")

        if validation_report.issues:
            print(f"\nIssues Found: {len(validation_report.issues)}")
            for issue in validation_report.issues[:5]:  # Show first 5
                print(f"  [{issue.severity.upper()}] {issue.message}")

        # Show text preview
        if result.get('raw_text'):
            text_preview = result['raw_text'][:200]
            print(f"\nText Preview (first 200 chars):")
            print(f"  {text_preview}...")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    print("\nWorkpedia Document Parser Demo")
    print("Phase 2: Document Processing Foundation\n")

    # Check for documents in data/input/
    input_dir = Path("data/input")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("\nTo test the parser, add documents to the data/input/ directory:")
        print("  - PDF files (.pdf)")
        print("  - Word documents (.docx)")
        print("  - HTML files (.html)")
        print("  - Images (.png, .jpg, .jpeg, .tiff)")
        print("\nExample:")
        print("  cp /path/to/your/document.pdf data/input/")
        return

    # Find supported documents
    supported_extensions = [
        ".pdf", ".docx", ".doc", ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".tiff", ".tif"
    ]

    documents = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not documents:
        print(f"No supported documents found in {input_dir}")
        print(f"\nSupported formats: {', '.join(supported_extensions)}")
        return

    print(f"Found {len(documents)} document(s):\n")
    for idx, doc in enumerate(documents, 1):
        print(f"  {idx}. {doc.name} ({doc.stat().st_size / 1024:.1f} KB)")

    # Process each document
    print(f"\n{'='*70}")
    print("Starting document processing...")
    print(f"{'='*70}")

    for doc in documents:
        process_document(doc)

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
