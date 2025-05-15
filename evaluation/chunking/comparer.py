import argparse
import json
import random
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.src.retrieval.components.chunkers.hybrid_chunker import HybridChunker
from backend.src.retrieval.components.chunkers.recursive_chunker import RecursiveChunker
from backend.src.retrieval.components.chunkers.semantic_chunker import SemanticChunker
from backend.src.retrieval.components.chunkers.simple_chunker import SimpleChunker
from backend.src.retrieval.document import Document


def write_results_to_file(
    results: Dict[str, Dict[str, Union[float, int]]],
    output_path: Union[str, Path],
    sample_results: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> None:
    """Write benchmark results and sample chunks to a file"""
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Chunker Benchmark Results\n\n## Summary\n\n")
        f.write(
            "| Chunker | Time (s) | Total Chunks | Avg Chunks/Doc | Avg Chunk Size | Std Dev Chunk Size |\n"
        )
        f.write(
            "|---------|----------|--------------|----------------|----------------|--------------------|\n"
        )
        for name, stats in results.items():
            f.write(
                f"| {name} | {stats['time_taken']:.2f} | {stats['total_chunks']} | {stats['avg_chunks_per_doc']:.2f} | {stats['avg_chunk_size']:.2f} | {stats['std_dev_chunk_size']:.2f} |\n"
            )

        f.write("\n## Detailed Results\n\n")
        for name, stats in results.items():
            f.write(
                f"### {name}\n\n- Processing time: {stats['time_taken']:.2f} seconds\n"
            )
            f.write(f"- Total chunks: {stats['total_chunks']}\n")
            f.write(
                f"- Chunk size standard deviation: {stats['std_dev_chunk_size']:.2f}\n\n"
            )

        if sample_results:
            f.write("\n## Sample Chunks\n\n")
            for chunker_name, docs in sample_results.items():
                f.write(f"### {chunker_name} Chunker\n\n")
                for doc_uuid, chunks in docs.items():
                    f.write(f"- **Document UUID**: {doc_uuid}\n  - Chunks:\n")
                    for i, chunk in enumerate(chunks, 1):
                        f.write(f"    {i}. {chunk}\n")
                    f.write("\n")


def load_documents(documents_path: Union[str, Path]) -> List[Document]:
    """Load documents from JSON files"""
    if isinstance(documents_path, str):
        documents_path = Path(documents_path)

    documents = []
    file_paths = sorted(documents_path.glob("*.json"))

    print(f"Found {len(file_paths)} document(s) in {documents_path}")

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            try:
                create_date = datetime.strptime(
                    doc_data["create_date"], "%m/%d/%Y %I:%M %p UTC"
                )
            except ValueError:
                create_date = datetime.now()

            doc = Document(
                vws_id=doc_data.get("vws_id", ""),
                uuid=uuid.UUID(doc_data.get("uuid", str(uuid.uuid4()))),
                create_date=create_date,
                content=doc_data.get("content", ""),
                link=doc_data.get("link", ""),
                attachment_links=doc_data.get("attachment_links", []),
                document_type=doc_data.get("type", None),
            )

            documents.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents


def process_sample_docs(
    chunkers: Dict[
        str, Union[SimpleChunker, RecursiveChunker, SemanticChunker, HybridChunker]
    ],
    sample_docs: List[Document],
) -> Dict[str, Dict[str, List[str]]]:
    """Process sample documents and collect chunks"""
    sample_results = {}

    for chunker_name, chunker in chunkers.items():
        sample_results[chunker_name] = {}
        last_subject = None

        for doc in sample_docs:
            try:
                # Process document
                if isinstance(chunker, (HybridChunker, SemanticChunker)):
                    chunks, _ = chunker.chunk_document(doc, last_subject)
                else:
                    chunks = chunker.chunk_document(doc)
                    if isinstance(chunks, tuple):
                        chunks, _ = chunks

                # Extract chunk contents
                chunk_contents = [chunk.content for chunk in chunks.values()]
                sample_results[chunker_name][str(doc.document_uuid)] = chunk_contents
            except Exception as e:
                print(
                    f"Error processing document {doc.document_uuid} with {chunker_name}: {e}"
                )

    return sample_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark document chunkers")
    parser.add_argument(
        "--documents", type=str, default="data/vws", help="Path to documents directory"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.md", help="Path to output file"
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    if not documents:
        print("No documents found. Exiting.")
        return

    # Initialize chunkers
    simple = SimpleChunker(1000)
    # semantic = SemanticChunker(
    #     model_name="all-MiniLM-L6-v2", context_size=3, similarity_threshold=0.7
    # )
    recursive = RecursiveChunker(chunk_size=1000, chunk_overlap=20, keep_separator=True)

    # short_semantic_chunker = SemanticChunker(
    #     model_name="paraphrase-MiniLM-L3-v2", context_size=2, similarity_threshold=0.6
    # )
    # long_semantic_chunker = SemanticChunker(
    #     model_name="multi-qa-mpnet-base-dot-v1",
    #     context_size=4,
    #     similarity_threshold=0.85,
    # )
    # hybrid = HybridChunker(
    #     short_semantic_chunker=short_semantic_chunker,
    #     standard_semantic_chunker=semantic,
    #     long_semantic_chunker=long_semantic_chunker,
    #     recursive_chunker=recursive,
    # )

    chunkers = {
        "simple": simple,
        "Recursive": recursive,
        # "Semantic": semantic,
        # "Hybrid": hybrid,
    }

    # Benchmark all documents
    results = benchmark_chunkers(documents, chunkers)

    # Process sample documents
    sample_docs = random.sample(documents, min(3, len(documents))) if documents else []
    sample_results = process_sample_docs(chunkers, sample_docs)

    write_results_to_file(results, args.output, sample_results)
    print("Results saved with sample chunks.")


def benchmark_chunkers(
    documents: List[Document],
    chunkers: Dict[str, Union[RecursiveChunker, SemanticChunker, HybridChunker]],
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Benchmark chunkers with type checking"""
    results = {}

    for name, chunker in chunkers.items():
        print("Benchmarking the", name, "chunker")
        total_chunks = 0
        chunk_sizes = []
        chunks_per_doc = []
        start_time = time.time()
        last_subject = None

        for doc in documents:
            try:
                if isinstance(chunker, (HybridChunker, SemanticChunker)):
                    chunks, last_subject = chunker.chunk_document(doc, last_subject)
                else:
                    chunks = chunker.chunk_document(doc)
                    if isinstance(chunks, tuple):
                        chunks, last_subject = chunks

                chunks_per_doc.append(len(chunks))
                total_chunks += len(chunks)
                chunk_sizes.extend(len(chunk.content) for chunk in chunks.values())
            except Exception as e:
                print(f"Error processing document {doc.document_uuid} with {name}: {e}")

        processing_time = time.time() - start_time
        avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
        std_dev = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0

        results[name] = {
            "time_taken": processing_time,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": (
                statistics.mean(chunks_per_doc) if chunks_per_doc else 0
            ),
            "avg_chunk_size": avg_chunk_size,
            "std_dev_chunk_size": std_dev,
        }

    return results


if __name__ == "__main__":
    main()
