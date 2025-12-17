"""Streamlit Web UI for Workpedia RAG System."""

import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from core.parser import DocumentParser
from core.query_engine import QueryEngine
from storage.vector_store import DocumentIndexer
from storage.history_store import HistoryStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Workpedia - RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .stat-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "query_engine" not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        try:
            # Step 1: Validate Ollama connectivity
            from core.llm import OllamaClient

            logger.info("Checking Ollama connectivity...")

            ollama_client = OllamaClient()
            health = ollama_client.health_check()

            if not health["server_reachable"]:
                st.error("❌ **Ollama Server Not Reachable**")
                st.error(health["message"])
                st.info(
                    """
                **To fix this:**
                1. Start Ollama: `ollama serve`
                2. Verify it's running: `ollama list`
                3. Check the URL in config/config.py
                """
                )
                st.stop()

            if not health["model_available"]:
                st.error("❌ **Model Not Available**")
                st.error(health["message"])
                st.info(
                    f"""
                **To fix this:**
                1. Pull the model: `ollama pull {health['model_name']}`
                2. Or use a different model in config/config.py

                **Available models:** {
                    ', '.join(health['available_models'])
                    if health['available_models']
                    else 'none'
                }
                """
                )
                st.stop()

            logger.info(f"✓ Ollama validated: {health['message']}")

            # Step 2: Initialize history store
            st.session_state.history_store = HistoryStore()
            logger.info("✓ History store initialized")

            # Step 3: Initialize components
            st.session_state.query_engine = QueryEngine(
                history_store=st.session_state.history_store,
                auto_save_history=True,
            )
            st.session_state.indexer = DocumentIndexer(
                vector_store=st.session_state.query_engine.vector_store,
                embedder=st.session_state.query_engine.embedder,
            )
            st.session_state.parser = DocumentParser()

            logger.info("✓ Streamlit app initialized successfully")

        except Exception as e:
            st.error("❌ **Initialization Failed**")
            st.error(f"Error: {e}")
            logger.error(f"Failed to initialize Streamlit app: {e}", exc_info=True)
            st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# Generate session ID for this browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# Header
st.markdown('<p class="main-header">📚 Workpedia</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Privacy-First RAG Document Question-Answering System</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # System Status
    st.subheader("System Status")
    try:
        health = st.session_state.query_engine.health_check()

        # LLM Status
        llm_status = health["llm"]["status"]
        llm_icon = "✅" if llm_status == "ok" else "❌"
        st.markdown(f"{llm_icon} **LLM**: {llm_status}")
        st.caption(f"Model: {health['llm']['model']}")

        # Vector Store Status
        st.markdown(f"✅ **Vector Store**: {health['vector_store']['status']}")
        st.caption(f"Documents: {health['vector_store']['documents']}")
        st.caption(f"Chunks: {health['vector_store']['chunks']}")

        # Embedder Status
        st.markdown(f"✅ **Embedder**: {health['embedder']['status']}")
        st.caption(f"Model: {health['embedder']['model']}")

    except Exception as e:
        st.error(f"System check failed: {e}")

    st.divider()

    # Query Settings
    st.subheader("Query Settings")
    n_results = st.slider(
        "Context chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve for context",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="LLM creativity (0=focused, 1=creative)",
    )

    st.divider()

    # Document Management
    st.subheader("📄 Indexed Documents")
    docs = st.session_state.query_engine.vector_store.list_documents()

    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc['filename'][:30]}..."):
                st.caption(f"**Chunks**: {doc['chunk_count']}")
                st.caption(f"**Doc ID**: {doc['doc_id'][:16]}...")

                if st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}"):
                    try:
                        st.session_state.query_engine.vector_store.delete_by_doc_id(doc["doc_id"])
                        st.success("Document deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
    else:
        st.info("No documents indexed yet")

    st.divider()

    # About
    with st.expander("ℹ️ About"):
        st.markdown(
            """
        **Workpedia** is a privacy-focused RAG system that:
        - Processes complex documents locally
        - Uses Ollama for LLM generation
        - Stores embeddings in ChromaDB
        - No data sent to external APIs

        Built with: Docling, Mistral, ChromaDB, FastAPI
        """
        )


# Main content area - Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat",
    "📤 Upload Documents",
    "📋 History",
    "⭐ Bookmarks",
    "📊 Statistics"
])

# Tab 1: Chat Interface
with tab1:
    st.header("Ask Questions About Your Documents")

    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            if chat.get("sources"):
                with st.expander(f"📑 Sources ({len(chat['sources'])} chunks)"):
                    for j, src in enumerate(chat["sources"], 1):
                        st.markdown(
                            f"**Source {j}** - {src['metadata'].get('filename', 'Unknown')}"
                        )
                        st.caption(f"Section: {src['metadata'].get('section', 'N/A')}")
                        st.caption(f"Similarity: {src['similarity']:.2%}")
                        st.text(
                            src["content"][:200] + "..."
                            if len(src["content"]) > 200
                            else src["content"]
                        )
                        st.divider()

            # Bookmark button
            if chat.get("query_id"):
                col1, col2 = st.columns([1, 9])
                with col1:
                    if st.button("⭐", key=f"bookmark_{i}", help="Bookmark this Q&A"):
                        try:
                            bookmark_id = st.session_state.history_store.add_bookmark(
                                query_id=chat["query_id"],
                                notes="",
                                tags=[],
                            )
                            st.success("Bookmarked!", icon="⭐")
                        except Exception as e:
                            st.error(f"Failed to bookmark: {e}")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Check if documents are indexed
        if not docs:
            st.warning("⚠️ No documents indexed! Please upload documents first.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.query_engine.query(
                            question=question,
                            n_results=n_results,
                            temperature=temperature,
                            session_id=st.session_state.session_id,
                            save_to_history=True,
                        )

                        st.write(result.answer)

                        # Show sources
                        if result.sources:
                            with st.expander(f"📑 Sources ({len(result.sources)} chunks)"):
                                for i, src in enumerate(result.sources, 1):
                                    filename = src["metadata"].get("filename", "Unknown")
                                    st.markdown(f"**Source {i}** - {filename}")
                                    st.caption(f"Section: {src['metadata'].get('section', 'N/A')}")
                                    st.caption(f"Similarity: {src['similarity']:.2%}")
                                    st.text(
                                        src["content"][:200] + "..."
                                        if len(src["content"]) > 200
                                        else src["content"]
                                    )
                                    st.divider()

                        # Save to history
                        st.session_state.chat_history.append(
                            {
                                "question": question,
                                "answer": result.answer,
                                "sources": result.sources,
                                "query_id": result.metadata.get("query_id"),
                            }
                        )

                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        logger.error(f"Query error: {e}", exc_info=True)

# Tab 2: Upload Documents
with tab2:
    st.header("Upload and Index Documents")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or HTML files",
            type=["pdf", "docx", "html", "htm"],
            accept_multiple_files=True,
            help="Upload documents to index them for question-answering",
        )

        if uploaded_files:
            if st.button("📥 Index Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")

                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(uploaded_file.name).suffix
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # Parse document
                        parsed_doc = st.session_state.parser.parse(tmp_path)

                        # Override metadata with original filename (not temp path)
                        if "metadata" in parsed_doc:
                            parsed_doc["metadata"]["filename"] = uploaded_file.name
                            parsed_doc["metadata"]["file_path"] = uploaded_file.name

                        # Index document
                        result = st.session_state.indexer.index_document(parsed_doc)

                        # Clean up temp file
                        Path(tmp_path).unlink()

                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)

                        # Show success
                        st.success(
                            f"✅ {uploaded_file.name}: {result['chunks_added']} chunks indexed"
                        )

                        # Track uploaded
                        st.session_state.uploaded_docs.append(
                            {
                                "filename": uploaded_file.name,
                                "chunks": result["chunks_added"],
                                "doc_id": result["doc_id"],
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")
                        logger.error(f"Indexing error for {uploaded_file.name}: {e}", exc_info=True)

                status_text.text("✅ All documents processed!")
                st.balloons()

                # Refresh sidebar
                st.rerun()

    with col2:
        st.info(
            """
        **Supported Formats:**
        - PDF
        - DOCX
        - HTML

        **What happens:**
        1. Parse document structure
        2. Extract text and tables
        3. Split into semantic chunks
        4. Generate embeddings
        5. Store in vector database
        """
        )

# Tab 3: History
with tab3:
    st.header("📋 Query History")
    st.markdown("Browse and search your past queries.")

    # Filters row
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        search_text = st.text_input("🔍 Search questions or answers", "", key="history_search")
    with col2:
        start_date = st.date_input("From date", value=None, key="history_start")
    with col3:
        end_date = st.date_input("To date", value=None, key="history_end")

    # Session filter and limit
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_session = st.checkbox("Current session only", key="history_filter_session")
    with col2:
        limit = st.number_input("Show", min_value=10, max_value=200, value=50, step=10, key="history_limit")

    st.divider()

    # Fetch queries
    try:
        if search_text:
            queries = st.session_state.history_store.search_queries(
                search_text=search_text,
                limit=limit,
            )
        else:
            queries = st.session_state.history_store.list_queries(
                limit=limit,
                session_id=st.session_state.session_id if filter_session else None,
                start_date=start_date.timestamp() if start_date else None,
                end_date=end_date.timestamp() if end_date else None,
            )

        st.info(f"Found {len(queries)} queries")

        # Display queries
        for query in queries:
            timestamp_str = datetime.fromtimestamp(query.timestamp).strftime('%Y-%m-%d %H:%M:%S')

            with st.expander(f"🕒 {timestamp_str} - {query.question[:60]}..."):
                st.markdown(f"**Question:** {query.question}")
                st.markdown(f"**Answer:**\n\n{query.answer}")

                # Sources
                if query.sources:
                    with st.expander(f"📚 Sources ({len(query.sources)})"):
                        for i, source in enumerate(query.sources, 1):
                            metadata = source.get("metadata", {})
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"- **File:** {metadata.get('filename', 'Unknown')}")
                            st.markdown(f"- **Section:** {metadata.get('section', 'N/A')}")
                            st.markdown(f"- **Similarity:** {source.get('similarity', 0):.2%}")
                            st.markdown(f"- **Content:** {source.get('content', '')[:200]}...")
                            st.markdown("---")

                # Metadata
                if query.metadata:
                    with st.expander("ℹ️ Metadata"):
                        st.json(query.metadata)

                # Actions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("⭐ Bookmark", key=f"hist_bm_{query.query_id}"):
                        try:
                            bookmark_id = st.session_state.history_store.add_bookmark(
                                query_id=query.query_id,
                                notes="",
                                tags=[],
                            )
                            st.success("Bookmarked!")
                        except Exception as e:
                            st.error(f"Error: {e}")

                with col2:
                    if st.button("🗑️ Delete", key=f"hist_del_{query.query_id}"):
                        st.session_state.history_store.delete_query(query.query_id)
                        st.success("Deleted!")
                        st.rerun()

                with col3:
                    md = st.session_state.history_store.export_query_markdown(query.query_id)
                    st.download_button(
                        "📄 Export MD",
                        md,
                        file_name=f"query_{query.query_id[:8]}.md",
                        mime="text/markdown",
                        key=f"hist_exp_{query.query_id}"
                    )

                with col4:
                    try:
                        pdf_bytes = st.session_state.history_store.export_queries_pdf([query.query_id])
                        st.download_button(
                            "📄 Export PDF",
                            pdf_bytes,
                            file_name=f"query_{query.query_id[:8]}.pdf",
                            mime="application/pdf",
                            key=f"hist_pdf_{query.query_id}"
                        )
                    except ImportError:
                        pass  # reportlab not installed

    except Exception as e:
        st.error(f"Error loading history: {e}")
        logger.error(f"History error: {e}", exc_info=True)

# Tab 4: Bookmarks
with tab4:
    st.header("⭐ Bookmarks")
    st.markdown("Your favorite Q&A pairs with notes and tags.")

    # Tag filter
    try:
        bookmarks = st.session_state.history_store.list_bookmarks(limit=1000)
        all_tags = set()
        for b in bookmarks:
            all_tags.update(b.tags)

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_tags = st.multiselect("🏷️ Filter by tags", sorted(all_tags), key="bookmark_tags")
        with col2:
            bm_limit = st.number_input("Show", min_value=10, max_value=200, value=50, step=10, key="bm_limit")

        # Filter bookmarks
        if selected_tags:
            filtered = st.session_state.history_store.list_bookmarks(
                tags=selected_tags,
                limit=bm_limit
            )
        else:
            filtered = bookmarks[:bm_limit]

        st.info(f"Found {len(filtered)} bookmarks")

        # Display bookmarks
        for bookmark in filtered:
            timestamp_str = datetime.fromtimestamp(bookmark.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            title = bookmark.query.question[:60] if bookmark.query else f"Bookmark {bookmark.bookmark_id[:8]}"

            with st.expander(f"⭐ {timestamp_str} - {title}..."):
                if bookmark.query:
                    st.markdown(f"**Question:** {bookmark.query.question}")
                    st.markdown(f"**Answer:**\n\n{bookmark.query.answer}")

                    # Sources
                    if bookmark.query.sources:
                        with st.expander(f"📚 Sources ({len(bookmark.query.sources)})"):
                            for i, source in enumerate(bookmark.query.sources, 1):
                                metadata = source.get("metadata", {})
                                st.markdown(f"**Source {i}:** {metadata.get('filename', 'Unknown')}")
                                st.markdown(f"- Section: {metadata.get('section', 'N/A')}")
                                st.markdown(f"- Similarity: {source.get('similarity', 0):.2%}")
                                st.markdown(f"- {source.get('content', '')[:150]}...")
                                st.markdown("---")

                st.markdown("---")

                # Editable notes and tags
                notes = st.text_area(
                    "📝 Notes",
                    value=bookmark.notes or "",
                    key=f"notes_{bookmark.bookmark_id}",
                    height=100
                )

                tags_input = st.text_input(
                    "🏷️ Tags (comma-separated)",
                    value=", ".join(bookmark.tags),
                    key=f"tags_{bookmark.bookmark_id}"
                )

                # Actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💾 Save Changes", key=f"save_{bookmark.bookmark_id}"):
                        new_tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                        success = st.session_state.history_store.update_bookmark(
                            bookmark.bookmark_id,
                            notes=notes,
                            tags=new_tags
                        )
                        if success:
                            st.success("Updated!")
                            st.rerun()
                        else:
                            st.error("Update failed")

                with col2:
                    if st.button("🗑️ Delete", key=f"del_{bookmark.bookmark_id}"):
                        success = st.session_state.history_store.delete_bookmark(bookmark.bookmark_id)
                        if success:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed")

                with col3:
                    if bookmark.query:
                        md = st.session_state.history_store.export_query_markdown(bookmark.query.query_id)
                        st.download_button(
                            "📄 Export",
                            md,
                            file_name=f"bookmark_{bookmark.bookmark_id[:8]}.md",
                            mime="text/markdown",
                            key=f"exp_{bookmark.bookmark_id}"
                        )

    except Exception as e:
        st.error(f"Error loading bookmarks: {e}")
        logger.error(f"Bookmark error: {e}", exc_info=True)

# Tab 5: Statistics
with tab5:
    st.header("System Statistics")

    try:
        stats = st.session_state.query_engine.vector_store.stats()

        # Stats overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Documents", stats["documents"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Chunks", stats["total_chunks"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            avg_chunks = stats["total_chunks"] / stats["documents"] if stats["documents"] > 0 else 0
            st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Document list with details
        st.subheader("Document Details")
        docs = st.session_state.query_engine.vector_store.list_documents()

        if docs:
            import pandas as pd

            df = pd.DataFrame(docs)
            st.dataframe(
                df[["filename", "chunk_count", "doc_id"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No documents indexed yet")

        st.divider()

        # System configuration
        st.subheader("Configuration")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**LLM Model**: {st.session_state.query_engine.llm.model}")
            st.markdown(f"**Embedding Model**: {st.session_state.query_engine.embedder.model_name}")

        with col2:
            st.markdown(
                f"**Embedding Dimension**: {st.session_state.query_engine.embedder.dimension}"
            )
            st.markdown(f"**Vector Store**: {stats['collection_name']}")

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")


# Footer
st.divider()
st.caption(
    "Workpedia - Privacy-First RAG System | "
    "All processing happens locally | "
    "No data sent to external APIs"
)
