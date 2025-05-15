import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Union

from backend.conf.config import Config
from backend.src.retrieval.components.chunkers.hybrid_chunker import HybridChunker
from backend.src.retrieval.components.chunkers.recursive_chunker import RecursiveChunker
from backend.src.retrieval.components.chunkers.semantic_chunker import SemanticChunker
from backend.src.retrieval.components.chunkers.simple_chunker import SimpleChunker
from backend.src.retrieval.components.embedders.sentence_transformer_ft import (
    SentenceTransformerFT,
)
from backend.src.retrieval.components.retrievers.chunk_retrievers.faiss_hnsw_retriever import (
    FaissRetriever,
)
from backend.src.retrieval.components.retrievers.document_retrievers.bm25_retriever import (
    BM25Retriever,
)
from backend.src.retrieval.components.tokenizers.spacy_tokenizer import SpacyTokenizer
from backend.src.retrieval.rag import RAGService


def run_chunking_comparison(
    chunkers: List[tuple],
    documents_path: Union[str, Path],
    output_base_path: Union[str, Path],
):
    """
    Compare multiple chunkers by processing the same set of documents.

    :param chunkers: List of tuples (chunker_name, chunker_instance)
    :param documents_path: Path to JSON documents
    :param output_base_path: Base output path for results
    """
    print("Starting Chunking Comparison:")
    comparison_results = []

    for chunker_name, chunker in chunkers:
        print(f"\nProcessing with {chunker_name}")

        # Create output path for this chunker
        output_path = Path(output_base_path) / chunker_name

        # Initialize RAGService with the current chunker
        tokenizer = SpacyTokenizer(model_name="nl_core_news_md", max_length=10_000_000)
        document_retriever = BM25Retriever(tokenizer=tokenizer)
        embedder = SentenceTransformerFT(
            model_name_or_path="all-MiniLM-L6-v2", device="cuda"
        ).half()
        chunk_retriever = FaissRetriever(embedding_model=embedder)

        retrieval_service = RAGService(
            tokenizer=tokenizer,
            chunker=chunker,
            document_retriever=document_retriever,
            chunk_retriever=chunk_retriever,
        )

        # Load and process documents
        start_time = time.time()
        for doc_path in Path(documents_path).glob("*.json"):
            retrieval_service.clean_content(doc_path)
        retrieval_service.index(docs_path=documents_path)
        end_time = time.time()
        total_time = end_time - start_time

        # Get chunked documents from RAGService
        chunked_documents = retrieval_service.chunked_documents

        # Collect statistics
        total_docs = len(chunked_documents)
        total_chunks = sum(len(doc.content) for doc in chunked_documents.values())
        chunk_sizes = [
            len(chunk.content)
            for doc in chunked_documents.values()
            for chunk in doc.content.values()
        ]

        # Compute chunk size statistics
        if chunk_sizes:
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            min_chunk_size = min(chunk_sizes)
            max_chunk_size = max(chunk_sizes)
        else:
            avg_chunk_size = min_chunk_size = max_chunk_size = 0

        # Store results
        result = {
            "chunker_name": chunker_name,
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "processing_time": total_time,
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
        }
        comparison_results.append(result)

        # Print summary for this chunker
        print(f"\n===== {chunker_name} SUMMARY =====")
        print(f"Documents processed: {total_docs}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average chunk size: {avg_chunk_size:.1f} characters")
        print(f"Min chunk size: {min_chunk_size} characters")
        print(f"Max chunk size: {max_chunk_size} characters")

        # Save output if needed
        if output_path:
            for doc_uuid, chunked_doc in chunked_documents.items():
                file_path = documents_path / f"{doc_uuid}.json"
                save_chunk_data(file_path, chunked_doc, output_path)

    # Print comparative summary
    print("\n===== CHUNKING COMPARISON =====")
    for result in comparison_results:
        print(f"\n{result['chunker_name']}:")
        print(f"  Total Docs: {result['total_docs']}")
        print(f"  Total Chunks: {result['total_chunks']}")
        print(f"  Processing Time: {result['processing_time']:.2f} seconds")
        print(f"  Avg Chunk Size: {result['avg_chunk_size']:.1f} characters")


def run_single_chunker(
    chunker,
    documents_path: Union[str, Path],
    output_path: Union[str, Path],
):
    """
    Run a single chunker on documents.

    :param chunker: Chunker instance to use
    :param documents_path: Path to JSON documents
    :param output_path: Output path for results
    """
    print("Running Single Chunker Analysis:")
    load_and_inspect_chunks(documents_path, chunker, output_path)


def load_and_inspect_chunks(
    documents_path: Union[str, Path],
    chunker,
    output_path: Union[str, Path] = None,
) -> None:
    """Load documents from JSON files, clean them, chunk them, and inspect the chunks."""
    documents_path = Path(documents_path)

    if output_path:
        output_path = Path(output_path)

        if output_path.exists():
            shutil.rmtree(output_path)  # Remove all existing files
        output_path.mkdir(parents=True)  # Create a fresh directory

    # Initialize RAGService
    tokenizer = SpacyTokenizer(model_name="nl_core_news_md", max_length=10_000_000)
    document_retriever = BM25Retriever(tokenizer=tokenizer)
    embedder = SentenceTransformerFT(
        model_name_or_path="all-MiniLM-L6-v2", device="cuda"
    ).half()
    chunk_retriever = FaissRetriever(embedding_model=embedder)

    retrieval_service = RAGService(
        tokenizer=tokenizer,
        chunker=chunker,
        document_retriever=document_retriever,
        chunk_retriever=chunk_retriever,
    )

    # STEP 1: Clean documents **before** processing
    cleaned_documents = {}
    for doc_file in documents_path.glob("*.json"):

        # If it prints "None", the method is missing or not initialized.
        cleaned_content = retrieval_service.clean_content(doc_file)
        cleaned_documents[doc_file.stem] = cleaned_content  # Store cleaned content

    # STEP 2: Process cleaned documents
    retrieval_service.index(docs_path=documents_path)
    # Get chunked documents
    chunked_documents = retrieval_service.chunked_documents
    total_docs = len(chunked_documents)
    total_chunks = sum(len(doc.content) for doc in chunked_documents.values())
    chunk_sizes = [
        len(chunk.content)
        for doc in chunked_documents.values()
        for chunk in doc.content.values()
    ]

    # Print summary statistics
    if total_docs > 0:
        print(f"Processed {total_docs} documents")
        print(f"Created {total_chunks} chunks")
        if chunk_sizes:
            print(
                f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters"
            )
            print(f"Min chunk size: {min(chunk_sizes)} characters")
            print(f"Max chunk size: {max(chunk_sizes)} characters")

    # Save output if needed
    if output_path:
        for doc_uuid, chunked_doc in chunked_documents.items():
            file_path = documents_path / f"{doc_uuid}.json"
            save_chunk_data(file_path, chunked_doc, output_path)


def save_chunk_data(file_path, chunked_doc, output_path):
    """Save chunk data to output files."""
    doc_output_dir = output_path / file_path.stem
    doc_output_dir.mkdir(exist_ok=True)

    # Write overview
    with open(doc_output_dir / "overview.txt", "w", encoding="utf-8") as f:
        f.write(f"Document: {file_path.name}\n")
        f.write(f"Document length: {len(chunked_doc.content)} characters\n")
        f.write(f"Total chunks: {len(chunked_doc.content)}\n\n")

        for i, (chunk_id, chunk) in enumerate(chunked_doc.content.items()):
            preview = chunk.content[:100].replace("\n", "\\n")
            f.write(f"Chunk {i + 1} (ID: {chunk_id})\n")
            f.write(f"Length: {len(chunk.content)} characters\n")
            f.write(f"Subject: {chunk.subject}\n")
            f.write(f"First Mentioned Date: {chunk.first_mentioned_date}\n")
            f.write(f"Last Mentioned Date: {chunk.last_mentioned_date}\n")
            f.write(f"Preview: {preview}...\n\n")

    # Save individual chunks
    chunks_dir = doc_output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    for i, (chunk_id, chunk) in enumerate(chunked_doc.content.items()):
        with open(chunks_dir / f"chunk_{i + 1}.txt", "w", encoding="utf-8") as f:
            f.write(chunk.content)

    # Create JSON summary
    chunks_data = {
        str(chunk_id): {
            "length": len(chunk.content),
            "content": (
                chunk.content[:200] + "..."
                if len(chunk.content) > 200
                else chunk.content
            ),
            "first_mentioned_date": (
                chunk.first_mentioned_date.isoformat()
                if hasattr(chunk, "first_mentioned_date") and chunk.first_mentioned_date
                else None
            ),
            "last_mentioned_date": (
                chunk.last_mentioned_date.isoformat()
                if hasattr(chunk, "last_mentioned_date") and chunk.last_mentioned_date
                else None
            ),
            "subject": chunk.subject,
            "document_type": chunk.document_type,
        }
        for chunk_id, chunk in chunked_doc.content.items()
    }

    with open(doc_output_dir / "chunks_summary.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)


def main():
    clean_path = "data/clean_vws"
    # Initialize cleaner.
    ragservice = RAGService()
    result = ragservice.run_parser("data/vws", clean_path)
    print(result)  # Output the result
    # Initialize chunkers
    simple_chunker = SimpleChunker(1000)

    semantic_chunker = SemanticChunker(
        context_size=Config.MEDIUM_CONTEXT_SIZE,
        similarity_threshold=Config.MEDIUM_SIMILARITY_THRES,
    )
    recursive_chunker = RecursiveChunker(
        chunk_size=1000, chunk_overlap=20, keep_separator=True
    )

    hybrid_chunker = HybridChunker(
        semantic_chunker=semantic_chunker,
        recursive_chunker=recursive_chunker,
    )

    # Chunkers for comparison
    chunkers_to_compare = [
        ("Simple", simple_chunker),
        ("Semantic", semantic_chunker),
        ("Recursive", recursive_chunker),
        ("Hybrid", hybrid_chunker),
    ]

    # Available modes
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspection.py compare - Compare all chunkers")
        print("  python inspection.py single <chunker_name>")
        print("\nAvailable chunkers:")
        for name, _ in chunkers_to_compare:
            print(f"  - {name}")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "compare":
        run_chunking_comparison(chunkers_to_compare, clean_path, "chunks/comparison")
    elif mode == "single":
        if len(sys.argv) < 3:
            print("Please specify a chunker name")
            sys.exit(1)

        chunker_name = sys.argv[2]

        # Find the matching chunker
        selected_chunker = next(
            (
                chunker
                for name, chunker in chunkers_to_compare
                if name.lower() == chunker_name.lower()
            ),
            None,
        )

        if selected_chunker:
            run_single_chunker(
                selected_chunker, clean_path, f"chunks/{chunker_name.lower()}"
            )
        else:
            print(f"Chunker '{chunker_name}' not found")
            sys.exit(1)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
