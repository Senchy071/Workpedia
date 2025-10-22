# workpedia

Property accounting and supply

## Setup

1. Activate the environment:
   ```bash
   source /media/production/projects/rag-workspace/venv/bin/activate
   ```

2. Add your documents to the `documents/` directory

3. Run the app:
   ```bash
   cd /media/production/projects/rag-workspace/projects/workpedia
   python app.py
   ```

## Project Structure

- `documents/` - Put your source documents here (.txt files)
- `vector_db/` - ChromaDB storage (auto-generated)
- `app.py` - Main application
- `config.py` - Configuration settings
- `logs/` - Application logs (if needed)

## Configuration

Edit `config.py` to change:
- Embedding model
- LLM model
- Chunk size
- Number of retrieval results

## Next Steps

1. Customize `app.py` for your specific use case
2. Adjust chunking strategy in `load_documents()`
3. Modify the prompt in `query()`
4. Add your own document processing logic
