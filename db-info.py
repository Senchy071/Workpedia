"""
Inspect what's in the vector database
"""

from pathlib import Path
import chromadb

# Connect to database
project_dir = Path(__file__).parent
db_path = project_dir / "vector_db"

client = chromadb.PersistentClient(path=str(db_path))

# Get collection
try:
    collection = client.get_collection(name="workpedia_knowledge")
    
    print("="*60)
    print("DATABASE CONTENTS")
    print("="*60)
    print(f"\nTotal chunks: {collection.count()}")
    
    # Get all chunks
    results = collection.get()
    
    # Group by document
    docs = {}
    for meta in results['metadatas']:
        source = meta['source']
        if source not in docs:
            docs[source] = 0
        docs[source] += 1
    
    print(f"\nDocuments indexed: {len(docs)}")
    print("\nBreakdown:")
    for doc, count in sorted(docs.items()):
        print(f"  {doc}: {count} chunks")
    
    print("\n" + "="*60)
    
except Exception as e:
    print(f"Error: {e}")
    print("Collection might not exist yet. Index some documents first.")
