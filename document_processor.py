"""
Structure-aware document processor for NATO/Military documents
Handles headers, references, chapters, and numbered sections
"""

from pypdf import PdfReader
from typing import List, Dict, Tuple
import re

class StructuredDocumentProcessor:
    """Process structured military documents with proper chunking"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.reader = PdfReader(pdf_path)
        
    def extract_all_pages(self) -> List[str]:
        """Extract text from all pages"""
        pages = []
        for page in self.reader.pages:
            text = page.extract_text()
            pages.append(text)
        return pages
    
    def clean_text(self, text: str) -> str:
        """Remove headers, footers, and boilerplate"""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip common headers/footers
            if 'NATO UNCLASSIFIED' in line_stripped:
                continue
            if 'Releasable to PfP' in line_stripped:
                continue
            if re.match(r'^\s*[-–—]?\s*\d+\s*[-–—]?\s*$', line_stripped):
                continue
            if re.match(r'^\s*[Pp]age\s+\d+', line_stripped):
                continue
            if 'AD 060-080' == line_stripped:
                continue
                
            # Skip empty lines
            if not line_stripped:
                continue
                
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def is_reference_line(self, line: str) -> bool:
        """Check if line is a reference (A., B., C., etc.)"""
        return bool(re.match(r'^[A-Z]\.\s+C-M\(', line))
    
    def is_toc_line(self, line: str) -> bool:
        """Detect Table of Contents lines"""
        # TOC lines have text followed by page numbers
        if re.search(r'.{20,}\s+\d{1,2}\s+\d+-\d+\s*$', line):
            return True
        if re.search(r'.{15,}\s+\d{1,3}\s+\d+\s*$', line):
            return True
        return False
    
    def detect_section_type(self, line: str) -> Tuple[str, int, str]:
        """
        Detect section type and level
        Returns (type, level, text)
        
        Priority order:
        1. CHAPTER X - TITLE
        2. X-Y. Numbered sections (like 3-8.)
        3. Letter subsections (a., b., c.)
        4. Numbered subsections ((1), (2))
        5. Top-level paragraphs (1., 2.)
        """
        line = line.strip()
        
        # CHAPTER X - TITLE (Level 1)
        if re.match(r'^CHAPTER\s+\d+', line, re.IGNORECASE):
            return 'chapter', 1, line
        
        # X-Y. Title (Level 2) - like "3-8. Redistribution"
        match = re.match(r'^(\d+-\d+)\.\s+([A-Z].{10,})$', line)
        if match:
            return 'numbered_section', 2, line
        
        # Letter subsection: a. title (Level 3)
        match = re.match(r'^([a-z])\.\s+(.{3,})$', line, re.IGNORECASE)
        if match and len(match.group(1)) == 1:
            # Make sure it's not just a reference marker
            if not re.match(r'^[a-z]\.\s+[A-Z]\.$', line):
                return 'letter', 3, line
        
        # Numbered subsection: (1) title (Level 4)
        match = re.match(r'^\((\d+)\)\s+(.{3,})$', line)
        if match:
            return 'numbered', 4, line
        
        # Top-level paragraph: 1. Title (Level 1)
        match = re.match(r'^(\d{1,2})\.\s+([A-Z][a-z].{10,})$', line)
        if match:
            # Skip if looks like TOC (ends with numbers)
            if re.search(r'\d{1,3}\s*$', line):
                return None, 0, None
            return 'paragraph', 1, line
        
        return None, 0, None
    
    def chunk_document(self) -> List[Dict]:
        """
        Main chunking logic with hierarchy tracking
        """
        pages = self.extract_all_pages()
        
        # Clean each page and combine
        cleaned_pages = [self.clean_text(p) for p in pages]
        full_text = '\n'.join(cleaned_pages)
        
        chunks = []
        current_chunk = []
        current_hierarchy = []
        in_references = False
        
        lines = full_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Handle REFERENCES section
            if 'REFERENCES:' in line:
                in_references = True
                if current_chunk:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'hierarchy': current_hierarchy.copy(),
                        'type': 'content'
                    })
                    current_chunk = []
                continue
            
            # Skip reference lines
            if in_references and self.is_reference_line(line):
                continue
            
            if in_references and not self.is_reference_line(line):
                in_references = False
            
            # Skip TOC lines
            if self.is_toc_line(line):
                continue
            
            # Detect section type
            section_type, level, section_text = self.detect_section_type(line)
            
            if section_type:
                # Save previous chunk
                if current_chunk and len('\n'.join(current_chunk).strip()) > 50:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'hierarchy': current_hierarchy.copy(),
                        'type': 'content'
                    })
                    current_chunk = []
                
                # Update hierarchy
                current_hierarchy = current_hierarchy[:level-1]
                if len(current_hierarchy) >= level:
                    current_hierarchy[level-1] = section_text
                else:
                    current_hierarchy.append(section_text)
                
                # Start new chunk with heading
                current_chunk = [section_text]
            else:
                # Regular content line
                current_chunk.append(line)
                
                # Create chunk if getting large (>1000 chars)
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > 1000:
                    # Look ahead - if next line is a section, break now
                    if i + 1 < len(lines):
                        next_type, _, _ = self.detect_section_type(lines[i+1].strip())
                        if next_type:
                            chunks.append({
                                'text': chunk_text,
                                'hierarchy': current_hierarchy.copy(),
                                'type': 'content'
                            })
                            current_chunk = []
        
        # Save final chunk
        if current_chunk and len('\n'.join(current_chunk).strip()) > 50:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'hierarchy': current_hierarchy.copy(),
                'type': 'content'
            })
        
        return chunks
    
    def format_for_rag(self, chunks: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Format chunks for RAG system with full context
        """
        documents = []
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(chunks):
            # Build context-aware text
            parts = []
            
            # Add hierarchy breadcrumb
            if chunk['hierarchy']:
                context = " → ".join(chunk['hierarchy'])
                parts.append(f"[SECTION: {context}]")
                parts.append("")
            
            # Add actual content
            parts.append(chunk['text'])
            
            full_text = '\n'.join(parts)
            
            # Extract section info for metadata
            section = chunk['hierarchy'][0] if chunk['hierarchy'] else 'Introduction'
            subsection = chunk['hierarchy'][1] if len(chunk['hierarchy']) > 1 else None
            
            documents.append(full_text)
            metadatas.append({
                'source': 'AD_060-080.pdf',
                'section': section,
                'subsection': subsection if subsection else '',
                'hierarchy_depth': len(chunk['hierarchy']),
                'type': chunk['type'],
                'chunk_id': idx
            })
            ids.append(f"doc_chunk_{idx}")
        
        return documents, metadatas, ids


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_file>")
        sys.exit(1)
    
    processor = StructuredDocumentProcessor(sys.argv[1])
    chunks = processor.chunk_document()
    
    print(f"\n{'='*60}")
    print(f"Processed {len(chunks)} chunks")
    print(f"{'='*60}\n")
    
    # Show sample chunks at different positions
    sample_indices = [0, 5, 10, 20, 30, 50]
    for i in sample_indices:
        if i < len(chunks):
            chunk = chunks[i]
            hier = ' → '.join(chunk['hierarchy']) if chunk['hierarchy'] else 'None'
            print(f"Chunk {i+1}:")
            print(f"  Hierarchy: {hier}")
            print(f"  Text (first 200 chars):")
            print(f"    {chunk['text'][:200]}")
            print()
