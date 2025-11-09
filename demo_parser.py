#!/usr/bin/env python3
"""Demo script for testing document parser functionality."""

import logging
import sys
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

    supported_extensions = [
        ".pdf", ".docx", ".doc", ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".tiff", ".tif"
    ]

    # Check if a file path was provided as command-line argument
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return

        if not input_path.is_file():
            print(f"Error: Not a file: {input_path}")
            return

        if input_path.suffix.lower() not in supported_extensions:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            return

        # Process single file
        print(f"Processing single file: {input_path.name} ({input_path.stat().st_size / (1024*1024):.2f} MB)\n")
        process_document(input_path)
        print("\nDemo complete!")
        return

    # No command-line argument, process directory
    input_dir = Path("data/input")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("\nUsage:")
        print(f"  {sys.argv[0]} <path-to-document>  # Process single file")
        print(f"  {sys.argv[0]}                     # Process all files in data/input/")
        print("\nTo test the parser, either:")
        print("  1. Provide a file path: python3 demo_parser.py /path/to/document.pdf")
        print("  2. Add documents to the data/input/ directory")
        print("\nSupported formats:")
        print("  - PDF files (.pdf)")
        print("  - Word documents (.docx)")
        print("  - HTML files (.html)")
        print("  - Images (.png, .jpg, .jpeg, .tiff)")
        return

    # Find supported documents
    documents = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not documents:
        print(f"No supported documents found in {input_dir}")
        print(f"\nSupported formats: {', '.join(supported_extensions)}")
        print(f"\nUsage: {sys.argv[0]} <path-to-document>")
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
