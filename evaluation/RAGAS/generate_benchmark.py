import argparse
import json
import os
import os.path
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import scrolledtext
from typing import Dict, List, Optional

import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Create document class equivalents for API mode
@dataclass
class APIDocumentChunk:
    uuid: str
    content: str
    content_date: str
    first_mentioned_date: Optional[str] = None
    last_mentioned_date: Optional[str] = None
    subject: Optional[str] = None
    document_type: Optional[str] = None


@dataclass
class APIChunkedDocument:
    uuid: str
    vws_id: str
    create_date: str
    content: Dict[str, APIDocumentChunk]
    link: str
    attachment_links: List[str]
    subject: Optional[str] = None
    document_type: Optional[str] = None


class APIClient:
    """Client for accessing the RAG API endpoints."""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def get_random_chunks(self, count=1):
        """Get random chunks from the API."""
        try:
            response = requests.get(
                f"{self.base_url}/random-chunks",
                params={"count": count},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            # Convert to APIDocumentChunk objects
            chunks = []
            for chunk_data in result.get("chunks", []):
                (
                    datetime.fromisoformat(chunk_data.get("content_date"))
                    if chunk_data.get("content_date")
                    else datetime.now()
                )
                chunk = APIDocumentChunk(
                    uuid=chunk_data.get("uuid"),
                    content=chunk_data.get("content", ""),
                    content_date=chunk_data.get("content_date"),
                    document_type=chunk_data.get("document_type"),
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Error fetching random chunks: {str(e)}")
            return []

    def get_random_documents(self, count=1):
        """Get random documents from the API."""
        try:
            response = requests.get(
                f"{self.base_url}/random-documents",
                params={"count": count},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            # Convert to APIChunkedDocument objects
            documents = []
            for doc_data in result.get("documents", []):
                doc_uuid = doc_data.get("uuid", "")
                content_dict = {}
                for chunk_id, chunk_data in doc_data.get("content", {}).items():
                    (
                        datetime.fromisoformat(chunk_data.get("content_date"))
                        if chunk_data.get("content_date")
                        else datetime.now()
                    )
                    chunk = APIDocumentChunk(
                        uuid=chunk_data.get("uuid"),
                        content=chunk_data.get("content", ""),
                        content_date=chunk_data.get("content_date"),
                        document_type=chunk_data.get("document_type"),
                    )
                    content_dict[chunk_id] = chunk

                doc = APIChunkedDocument(
                    uuid=doc_uuid,
                    vws_id=doc_data.get("vws_id", ""),
                    create_date=doc_data.get("create_date", ""),
                    content=content_dict,
                    link=doc_data.get("link", ""),
                    attachment_links=doc_data.get("attachment_links", []),
                    document_type=doc_data.get("document_type"),
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error fetching random documents: {str(e)}")
            return []

    def get_llm_response(self, prompt):
        """Get a direct response from the LLM via the API."""
        try:
            response = requests.post(
                f"{self.base_url}/direct-llm",
                json={"prompt": prompt},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return ""


def extract_final_response(
    text: str, start_marker: str = "<", end_marker: str = ">"
) -> str:
    """
    Extracts the content between start_marker and end_marker if present.
    If the markers are not found, returns the full trimmed text.
    """
    start = text.find(start_marker)
    end = text.rfind(end_marker)
    if start != -1 and end != -1 and end > start:
        return text[start + len(start_marker) : end].strip()
    return text.strip()


def generate_answer(api_client: APIClient, question: str, context: str) -> str:
    prompt = (
        "Gegeven de volgende context en vraag, genereer een volledig en correct antwoord op basis van de context.\n\n"
        f"Context:\n{context}\n\nVraag:\n{question}\n\n"
        "Plaats het uiteindelijke antwoord in het Nederlands."
    )
    raw_answer = api_client.get_llm_response(prompt)
    answer = extract_final_response(raw_answer)
    return answer


def generate_specific_benchmark(api_client: APIClient, data_provider) -> dict:
    chunks = data_provider.get_random_chunks(1)
    chosen_chunk = random.choice(chunks)
    original_text = chosen_chunk.content

    prompt = (
        "Onderstaande tekst is een fragment uit een document. "
        f"\n\n{original_text}\n\n"
        "Genereer een vraag die zich richt op de details en inhoud van het fragment. "
        "De vraag moet zo worden geformuleerd dat het antwoord in de tekst terug te vinden is. "
        "Vermijd het gebruik van placeholders of verwijzingen naar het fragment buiten de verstrekte tekst.\n"
        "Voorbeelden van goede vragen: "
        "Wat was het advies van het RIVM over thuiswerken tijdens de coronacrisis? "
        "Welke protocollen heeft NOC*NSF ontwikkeld voor het hervatten van sportactiviteiten? "
        "Voor hoeveel euro is de mondkapjesdeal in mei 2024 tot stand gekomen?\n"
        "Plaats de uiteindelijke vraag in het Nederlands."
    )
    raw_question = api_client.get_llm_response(prompt)
    question = extract_final_response(raw_question)
    answer = generate_answer(api_client, question, original_text)
    return {
        "type": "specific",
        "question": question,
        "answer": answer,
        "relevant_uuids": [str(chosen_chunk.document_uuid)],
        "context": original_text,
    }


def generate_document_benchmark(api_client: APIClient, data_provider) -> dict:
    document = None
    documents = data_provider.get_random_documents(1)
    if not documents:
        raise ValueError("No documents found.")
    document = documents[0]
    doc_chunks = list(document.content.values())
    full_text = "\n".join(chunk.content for chunk in doc_chunks)

    prompt = (
        "Onderstaande tekst betreft het volledige document, samengesteld uit meerdere delen. "
        f"Text: {full_text}"
        "Formuleer een heldere, algemene vraag die de kern en de belangrijkste elementen van dit document omvat. "
        "De vraag moet volledig gebaseerd zijn op de verstrekte tekst en ontworpen zijn om de effectiviteit van een retrieval-systeem te evalueren. "
        "Vermijd het gebruik van placeholders of expliciete verwijzingen naar de tekst zelf.\n"
        "Voorbeelden van goede vragen: "
        "Wat was het advies van het RIVM over thuiswerken tijdens de coronacrisis? "
        "Welke protocollen heeft NOC*NSF ontwikkeld voor het hervatten van sportactiviteiten? "
        "Voor hoeveel euro is de mondkapjesdeal tot stand gekomen? "
        "Hoelang duurde het meest recente kabinetsvorming onder leiding van Dick Schoof? "
        "Wie waren de vertegenwoordigers van Nederland tijdens de Olympische Spelen in Parijs 2024?\n"
        "Plaats de uiteindelijke vraag in het Nederlands."
    )
    raw_question = api_client.get_llm_response(prompt)
    question = extract_final_response(raw_question)
    answer = generate_answer(api_client, question, full_text)
    return {
        "type": "document",
        "question": question,
        "answer": answer,
        "relevant_uuids": [str(chunk.document_uuid) for chunk in doc_chunks],
        "context": full_text,
    }


def generate_multi_document_benchmark(
    api_client: APIClient, clusters: list, docs_per_benchmark: int
) -> dict:
    valid_clusters = [
        cluster for cluster in clusters if len(cluster) >= docs_per_benchmark
    ]
    if not valid_clusters:
        raise ValueError("No clusters with sufficient documents found")
    cluster = random.choice(valid_clusters)
    selected_docs = random.sample(cluster, docs_per_benchmark)
    summaries = []
    relevant_ids = []
    for doc in selected_docs:
        doc_chunks = list(doc.content.values())
        doc_text = "\n".join(chunk.content for chunk in doc_chunks)
        summaries.append(doc_text)
        relevant_ids.extend(str(chunk.document_uuid) for chunk in doc_chunks)
    combined_text = "\n\n=== Document Separation ===\n\n".join(summaries)

    prompt = (
        "Je bent een expert in het maken van test queries voor een RAG systeem. "
        "Gegeven de volgende text van meerdere documenten die inhoudelijk verwant zijn, formuleer je een samengestelde vraag. "
        f"Text: {combined_text}"
        "De vraag moet:"
        "\n- Informatie uit de verschillende documenten combineren,"
        "\n- Vergelijkende en synthetiserende elementen bevatten,"
        "\n- De onderlinge relaties tussen de documenten verkennen,"
        "\n- Specifiek en natuurlijk klinken, alsof een echte gebruiker de vraag stelt,"
        "\n- Een mix van feitelijke en interpretatieve elementen omvatten,"
        "\n- Zelfstandig leesbaar zijn zonder verdere context. "
        'Vermijd verwijzingen naar "dit document", "deze versie" of "de bijlage". Gebruik in plaats daarvan specifieke beschrijvingen. '
        "Plaats de uiteindelijke vraag in het Nederlands."
    )
    raw_question = api_client.get_llm_response(prompt)
    question = extract_final_response(raw_question)
    answer = generate_answer(api_client, question, combined_text)
    return {
        "type": "multi_document",
        "question": question,
        "answer": answer,
        "relevant_uuids": relevant_ids,
        "context": combined_text,
    }


def cluster_documents(data_provider, num_clusters: int = 10) -> list:
    documents = data_provider.get_random_documents(1000)
    if not documents:
        raise ValueError("No documents found for clustering")
    texts = []
    for doc in documents:
        chunks = list(doc.content.values())
        full_text = "\n".join(chunk.content for chunk in chunks)
        texts.append(full_text)
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = [[] for _ in range(num_clusters)]
    for doc, label in zip(documents, labels):
        clusters[label].append(doc)
    return [cluster for cluster in clusters if cluster]


def approve_candidate(benchmark: dict) -> dict:
    """
    Opens a Tkinter window to allow you to review and edit the generated benchmark.
    Returns the updated benchmark if approved, or None if rejected.
    """
    result = {}
    root = tk.Tk()
    root.title("Benchmark Review")
    root.attributes("-fullscreen", True)

    label_font = ("Helvetica", 20, "bold")
    text_font = ("Helvetica", 16)

    tk.Label(root, text="Benchmark Review", font=("Helvetica", 28, "bold")).pack(
        pady=20
    )
    tk.Label(
        root,
        text="Review the generated benchmark candidate below. Edit the question and answer if needed, then click 'Approve'. Click 'Reject' to generate a new candidate.",
        font=("Helvetica", 18),
        wraplength=1000,
        justify="center",
    ).pack(pady=10)

    tk.Label(root, text=f"Benchmark Type: {benchmark['type']}", font=label_font).pack(
        pady=10
    )

    # Editable Question
    tk.Label(root, text="Generated Question (editable):", font=label_font).pack(
        anchor="w", padx=20
    )
    question_box = scrolledtext.ScrolledText(
        root, height=10, width=120, font=text_font, wrap="word"
    )
    question_box.insert("1.0", benchmark["question"])
    question_box.pack(pady=10, padx=20)

    # Editable Answer
    tk.Label(root, text="Generated Answer (editable):", font=label_font).pack(
        anchor="w", padx=20
    )
    answer_box = scrolledtext.ScrolledText(
        root, height=10, width=120, font=text_font, wrap="word"
    )
    answer_box.insert("1.0", benchmark["answer"])
    answer_box.pack(pady=10, padx=20)

    # Read-only Context
    tk.Label(root, text="Context (read-only):", font=label_font).pack(
        anchor="w", padx=20
    )
    context_box = scrolledtext.ScrolledText(
        root, height=25, width=120, font=text_font, wrap="word"
    )
    context_box.insert("1.0", benchmark["context"])
    context_box.config(state="disabled")
    context_box.pack(pady=10, padx=20)

    def on_decision(decision):
        if decision:
            # Update benchmark with the edited question and answer
            benchmark["question"] = question_box.get("1.0", "end").strip()
            benchmark["answer"] = answer_box.get("1.0", "end").strip()
            result["approved"] = True
            result["benchmark"] = benchmark
        else:
            result["approved"] = False
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)
    tk.Button(
        btn_frame,
        text="Approve",
        command=lambda: on_decision(True),
        font=label_font,
        width=15,
    ).pack(side="left", padx=20)
    tk.Button(
        btn_frame,
        text="Reject",
        command=lambda: on_decision(False),
        font=label_font,
        width=15,
    ).pack(side="right", padx=20)

    root.mainloop()
    if result.get("approved"):
        return result.get("benchmark")
    return None


def generate_candidates_batch(generate_func, batch_size, **kwargs):
    """Generates a batch of candidates up front."""
    return [generate_func(**kwargs) for _ in range(batch_size)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmarks for a RAG system."
    )
    parser.add_argument(
        "--num-specific",
        type=int,
        default=1,
        help="Number of specific benchmarks to generate",
    )
    parser.add_argument(
        "--num-document",
        type=int,
        default=1,
        help="Number of document benchmarks to generate",
    )
    parser.add_argument(
        "--num-multi-document",
        type=int,
        default=1,
        help="Number of multi-document benchmarks to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of candidates to generate in each batch",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:5000",
        help="Base URL for API endpoints",
    )
    args = parser.parse_args()

    num_specific = args.num_specific
    num_document = args.num_document
    num_multi_document = args.num_multi_document
    batch_size = args.batch_size
    docs_per_multi = 3
    benchmarks = []

    # Initialize the API client
    api_url = args.api_url
    print(f"Using API at: {api_url}")
    api_client = APIClient(base_url=api_url)
    data_provider = api_client  # Use the same client for data and LLM

    # Cluster documents for multi-document benchmarks
    clusters = cluster_documents(data_provider, num_clusters=10)

    # Generate specific benchmarks
    print("Generating specific benchmarks...")
    count = 0
    while count < num_specific:
        batch = generate_candidates_batch(
            generate_specific_benchmark,
            batch_size,
            api_client=api_client,
            data_provider=data_provider,
        )
        for candidate in batch:
            reviewed_candidate = approve_candidate(candidate)
            if reviewed_candidate is not None:
                benchmarks.append(reviewed_candidate)
                count += 1
                if count >= num_specific:
                    break

    # Generate document benchmarks
    print("Generating document benchmarks...")
    count = 0
    while count < num_document:
        batch = generate_candidates_batch(
            generate_document_benchmark,
            batch_size,
            api_client=api_client,
            data_provider=data_provider,
        )
        for candidate in batch:
            reviewed_candidate = approve_candidate(candidate)
            if reviewed_candidate is not None:
                benchmarks.append(reviewed_candidate)
                count += 1
                if count >= num_document:
                    break

    # Generate multi-document benchmarks
    print("Generating multi-document benchmarks...")
    count = 0
    while count < num_multi_document:
        batch = generate_candidates_batch(
            generate_multi_document_benchmark,
            batch_size,
            api_client=api_client,
            clusters=clusters,
            docs_per_benchmark=docs_per_multi,
        )
        for candidate in batch:
            reviewed_candidate = approve_candidate(candidate)
            if reviewed_candidate is not None:
                benchmarks.append(reviewed_candidate)
                count += 1
                if count >= num_multi_document:
                    break

    with open("retrieval_benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=4)
    print("Benchmarks saved to retrieval_benchmarks.json")


if __name__ == "__main__":
    main()
