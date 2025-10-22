"""
Project Configuration
Customize these settings for your project
"""

CONFIG = {
    # Project info
    'project_name': 'workpedia',
    'description': 'Property accounting and supply',
    
    # Collection name for vector DB
    'collection_name': 'workpedia_knowledge',
    
    # Model settings
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'llm_model': 'qwen2.5:32b-instruct-q4_K_M',
    
    # Chunking settings
    'chunk_size': 1000,
    'chunk_overlap': 200,
    
    # Retrieval settings
    'n_results': 3,
}
