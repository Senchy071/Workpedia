"""
Manage vector database - delete documents
"""

from pathlib import Path
import chromadb
import sys

project_dir = Path(__file__).parent
db_path = project_dir / "vector_db"
client = chromadb.PersistentClient(path=str(db_path))

def list_documents():
    """Show all indexed documents"""
    try:
        collection = client.get_collection(name="workpedia_knowledge")
        results = collection.get()
        
        docs = {}
        for meta in results['metadatas']:
            source = meta['source']
            if source not in docs:
                docs[source] = 0
            docs[source] += 1
        
        print("\nIndexed documents:")
        for i, (doc, count) in enumerate(sorted(docs.items()), 1):
            print(f"  {i}. {doc} ({count} chunks)")
        
        return sorted(docs.keys())
    except:
        print("No documents indexed yet.")
        return []

def delete_document(doc_name):
    """Delete all chunks from a specific document"""
    try:
        collection = client.get_collection(name="workpedia_knowledge")
        results = collection.get()
        
        # Find IDs of chunks from this document
        ids_to_delete = [
            id for id, meta in zip(results['ids'], results['metadatas'])
            if meta['source'] == doc_name
        ]
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"\n✓ Deleted {len(ids_to_delete)} chunks from '{doc_name}'")
        else:
            print(f"\nDocument '{doc_name}' not found in database.")
    
    except Exception as e:
        print(f"Error: {e}")

def delete_all():
    """Delete everything and start fresh"""
    try:
        client.delete_collection(name="workpedia_knowledge")
        print("\n✓ Deleted entire collection. Run app.py to create fresh.")
    except:
        print("Collection doesn't exist.")

# Main menu
if __name__ == "__main__":
    print("="*60)
    print("DATABASE MANAGEMENT")
    print("="*60)
    
    docs = list_documents()
    
    if not docs:
        sys.exit(0)
    
    print("\nOptions:")
    print("  1. Delete specific document")
    print("  2. Delete ALL and start fresh")
    print("  3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        doc_name = input("\nEnter document name to delete: ").strip()
        confirm = input(f"Delete '{doc_name}'? (y/N): ").strip().lower()
        if confirm == 'y':
            delete_document(doc_name)
    
    elif choice == "2":
        confirm = input("\nDelete EVERYTHING? (y/N): ").strip().lower()
        if confirm == 'y':
            delete_all()
    
    print()
