"""
RAG Application Template
Customize this for your specific use case
"""

import os
from pathlib import Path
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

class RAGApp:
    """Your custom RAG application"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.docs_dir = self.project_dir / "documents"
        self.db_path = self.project_dir / "vector_db"
        
        # Load configuration
        from config import CONFIG
        self.config = CONFIG
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.config['embedding_model']}...")
        self.embedder = SentenceTransformer(self.config['embedding_model'])
        
        # Initialize ChromaDB
        print("Connecting to vector database...")
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config['collection_name'],
            metadata={"description": self.config['description']}
        )
        
        print("✓ RAG App initialized!")
    
    def load_documents(self):
        """
        Load documents using structure-aware processor

        Returns:
            tuple: (documents, metadatas, ids) where:
                - documents: List of text chunks
                - metadatas: List of dicts with source, type, and hierarchy info
                - ids: List of unique chunk identifiers
        """
        from document_processor import StructuredDocumentProcessor
        from docx import Document as DocxDocument

        documents = []
        metadatas = []
        ids = []

        # Handle different file types with appropriate processing strategies
        file_count = 0

        # Process PDFs with structure-aware processor
        # Uses hierarchy detection for chapters/sections to preserve document structure
        for filepath in self.docs_dir.glob("*.pdf"):
            file_count += 1
            print(f"  Processing {filepath.name} with structure-aware parser...")
            
            processor = StructuredDocumentProcessor(str(filepath))
            chunks = processor.chunk_document()  # Detects chapters, sections, removes TOC/headers
            docs, metas, chunk_ids = processor.format_for_rag(chunks)  # Adds hierarchy breadcrumbs
            
            documents.extend(docs)
            metadatas.extend(metas)
            ids.extend(chunk_ids)
        
        # Process Word docs (simple paragraph-based chunking)
        # No structure detection - just splits on double newlines
        for filepath in self.docs_dir.glob("*.docx"):
            file_count += 1
            print(f"  Processing {filepath.name}...")

            doc = DocxDocument(filepath)
            content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": filepath.name,
                    "file_type": "docx",
                    "chunk_id": chunk_idx
                })
                ids.append(f"{filepath.stem}_chunk_{chunk_idx}")
        
        # Process text files (simple paragraph-based chunking)
        # Splits on double newlines - assumes paragraphs are separated by blank lines
        for filepath in self.docs_dir.glob("*.txt"):
            file_count += 1
            print(f"  Processing {filepath.name}...")

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": filepath.name,
                    "file_type": "text",
                    "chunk_id": chunk_idx
                })
                ids.append(f"{filepath.stem}_chunk_{chunk_idx}")
        
        print(f"  ✓ Loaded {len(documents)} chunks from {file_count} files")
        return documents, metadatas, ids
    
    def index_documents(self):
        """
        Index all documents in the documents directory

        Loads documents, generates embeddings, and stores in ChromaDB.
        Prompts for re-indexing if collection already exists.
        """
        print("\nIndexing documents...")

        # Check if already indexed - avoid duplicate indexing
        if self.collection.count() > 0:
            print(f"  Collection has {self.collection.count()} documents")
            response = input("  Re-index? (y/N): ").strip().lower()
            if response != 'y':
                return

            # Delete old collection and create fresh one for re-indexing
            # This ensures no duplicate or stale chunks remain
            self.chroma_client.delete_collection(self.config['collection_name'])
            self.collection = self.chroma_client.create_collection(
                name=self.config['collection_name'],
                metadata={"description": self.config['description']}
            )
        
        documents, metadatas, ids = self.load_documents()
        
        if not documents:
            print("  No documents found in documents/ directory")
            return
        
        print(f"  Loaded {len(documents)} chunks")
        print("  Generating embeddings...")
        # Convert text chunks to vector embeddings using sentence-transformers
        embeddings = self.embedder.encode(documents, show_progress_bar=True)

        print("  Storing in vector database...")
        # Store embeddings with their source text, metadata, and unique IDs
        # ChromaDB uses embeddings for similarity search, returns documents+metadata
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Indexed {len(documents)} chunks!")
    
    def query(self, question: str, n_results: int = 3):
        """
        Query the RAG system using retrieval + generation

        Args:
            question: User's question
            n_results: Number of chunks to retrieve (default: 3)

        Returns:
            str: Generated answer or None on error
        """
        print(f"\nQuery: {question}")

        # RETRIEVE: Find most relevant chunks using vector similarity
        query_embedding = self.embedder.encode([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        docs = results.get('documents')
        context_chunks = docs[0] if docs else []
        
        print(f"Retrieved {len(context_chunks)} chunks")
        print(f"\nChunks retrieved:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

        # GENERATE: Use LLM to answer question based on retrieved context
        context = "\n\n".join([f"Context {i+1}: {chunk}"
                               for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Based on the context, answer the question.
If unsure, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            # Send prompt with context to local Ollama LLM
            response = ollama.chat(
                model=self.config['llm_model'],
                messages=[{'role': 'user', 'content': prompt}]
            )
            answer = response['message']['content']
            print(f"\nAnswer: {answer}\n")
            return answer
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

def main():
    """Main entry point"""
    project_dir = Path(__file__).parent
    app = RAGApp(str(project_dir))
    
    # Index documents
    app.index_documents()
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                app.query(question)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
