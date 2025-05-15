"""Training data generation module.

This module provides functionality to generate training data for retrieval models
including embedders and rerankers. It uses document chunks and LLM query generation
to create training examples with positive and negative pairs.

The TrainingDataGenerator class handles clustering chunks, generating natural language
queries, and creating training data with appropriate positive/negative examples.
"""

import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List
from uuid import UUID

from sklearn.cluster import MiniBatchKMeans  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from tqdm import tqdm

from backend.src.data_classes import ChunkedDocument, ChunkID, DocumentChunk
from backend.src.services.llm import BaseLLMService
from backend.src.services.retrieval.components.document_retriever import (
    BaseDocumentRetriever,
)

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates training data for retrieval models.

    This class handles generation of training data for embedders and rerankers by:
    1. Clustering document chunks to ensure diverse training examples
    2. Using LLM to generate natural language queries
    3. Creating positive and negative examples using BM25 retrieval
    4. Verifying training examples with LLM

    Attributes:
        chunks: List of document chunks to generate training data from
        llm_service: LLM service for query generation and verification
        num_clusters: Number of clusters for document chunking
        bm25: BM25 retriever for finding hard negatives
        chunk_clusters: Dictionary mapping cluster IDs to lists of chunks
    """

    def __init__(
        self,
        documents: Dict[UUID, ChunkedDocument],
        chunks: Dict[ChunkID, DocumentChunk],
        document_retriever: BaseDocumentRetriever,
        llm_service: BaseLLMService,
        num_clusters: int = 10,
    ):
        """Initialize the training data generator.

        Args:
            documents: Dictionary of documents from RAG
            chunks: Dictionary of chunks from RAG
            bm25_retriever: BM25 retriever for finding hard negatives
            llm_service: LLM service for query generation
            num_clusters: Number of clusters for document clustering
        """
        # Only work with chunks, ignoring documents
        self.chunks = list(chunks.values())
        self.llm_service = llm_service
        self.num_clusters = num_clusters
        self.document_retriever = document_retriever

        # Create chunk clusters instead of document clusters
        self.chunk_clusters = self._cluster_chunks()

    def generate_chunk_query_pairs(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate query-chunk pairs for embedder training.

        Creates pairs of queries and chunks where the query is generated to be
        relevant to the chunk content.

        Args:
            num_samples: Number of query-chunk pairs to generate

        Returns:
            List of dictionaries containing query and chunk pairs
        """
        pairs: List[Dict[str, str]] = []
        for _ in tqdm(range(num_samples), desc="Generating chunk-query pairs"):
            # Randomly select a chunk
            chunk = random.choice(self.chunks)

            # Generate a query for this chunk
            query = self.generate_query(chunk.content)

            # Create a positive pair
            pairs.append(
                {
                    "query": query,
                    "chunk": chunk.content,
                }
            )

        return pairs

    def generate_reranker_chunk_pairs(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate query-chunk pairs for chunk reranker training.

        Creates pairs of queries and chunks with binary relevance labels.
        Uses BM25 to find hard negative examples.

        Args:
            num_samples: Number of query-chunk pairs to generate

        Returns:
            List of dictionaries containing query, chunk and relevance label
        """
        pairs: List[Dict[str, Any]] = []
        for _ in tqdm(range(num_samples), desc="Generating reranker chunk pairs"):
            # Randomly select a chunk
            pos_chunk = random.choice(self.chunks)

            # Generate a query for this chunk
            query = self.generate_query(pos_chunk.content)
            if not query:  # Skip if query generation failed
                continue

            # Find a negative chunk using BM25
            neg_chunks_result = self.document_retriever.retrieve([query], 10)
            if not neg_chunks_result or not neg_chunks_result[0]:
                continue

            neg_chunks = neg_chunks_result[0]
            # Convert retrieved content back to chunks
            neg_chunk_candidates = [
                chunk
                for chunk in self.chunks
                if chunk.content in [neg_chunk.content for neg_chunk in neg_chunks]
            ]
            neg_chunk = next(
                (
                    chunk
                    for chunk in neg_chunk_candidates
                    if chunk.uuid != pos_chunk.uuid
                ),
                None,
            )

            # If no suitable negative chunk found via BM25, randomly select one
            if not neg_chunk:
                neg_candidates = [c for c in self.chunks if c.uuid != pos_chunk.uuid]
                if neg_candidates:  # Only proceed if we have candidates
                    neg_chunk = random.choice(neg_candidates)
                else:
                    continue  # Skip this iteration if no negative chunk available

            # Add positive and negative pairs
            pairs.append({"query": query, "text": pos_chunk.content, "label": 1})
            pairs.append({"query": query, "text": neg_chunk.content, "label": 0})

        return pairs

    def generate_reranker_document_pairs(
        self, num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate query-chunk pairs for document-level reranker training.

        Similar to chunk-level pairs but intended for document reranking.
        Uses BM25 to find hard negative examples at the document level.

        Args:
            num_samples: Number of query-document pairs to generate

        Returns:
            List of dictionaries containing query, document and relevance label
        """
        pairs: List[Dict[str, Any]] = []

        for _ in tqdm(
            range(num_samples), desc="Generating reranker document-level pairs"
        ):
            # Randomly select a chunk to act as a positive example
            pos_chunk = random.choice(self.chunks)

            # Generate query from the chunk content
            query = self.generate_query(pos_chunk.content)
            if not query:  # Skip if query generation failed
                continue

            # Find negative chunk using BM25
            neg_chunks_result = self.document_retriever.retrieve([query], 10)
            if not neg_chunks_result or not neg_chunks_result[0]:
                continue

            neg_chunks_content = neg_chunks_result[0]

            # Find chunks that are different from the positive chunk
            neg_chunk_candidates = [
                chunk
                for chunk in self.chunks
                if any(
                    content in chunk.content
                    for content in [
                        neg_chunk.content for neg_chunk in neg_chunks_content
                    ]
                )
                and chunk.uuid != pos_chunk.uuid
            ]

            # If no suitable negative chunk found via BM25, randomly select one
            if not neg_chunk_candidates:
                neg_candidates = [c for c in self.chunks if c.uuid != pos_chunk.uuid]
                if neg_candidates:
                    neg_chunk = random.choice(neg_candidates)
                else:
                    continue
            else:
                neg_chunk = neg_chunk_candidates[0]

            # Add positive and negative pairs
            pairs.append({"query": query, "text": pos_chunk.content, "label": 1})
            pairs.append({"query": query, "text": neg_chunk.content, "label": 0})

        return pairs

    def generate_query(self, content: str) -> str:
        """Generate a query for a document chunk using LLM.

        Uses the LLM service to formulate a natural question that is relevant
        to the provided document chunk. The prompt ensures the query is realistic
        and ends with a question mark.

        Args:
            content: Text content of the document chunk

        Returns:
            str: Generated query for the chunk
        """
        messages = [
            {
                "role": "user",
                "content": f"""Gegeven deze document fragmenten:

            {content}

            Genereer een natuurlijke vraag die relevant is voor deze fragmenten.
            De vraag moet:
            1. Eindigen met een vraagteken
            2. Specifiek genoeg zijn om deze fragmenten te vinden
            3. Algemeen genoeg zijn om realistisch te zijn
            4. Een echte vraag zijn die iemand zou kunnen stellen
            5. Géén verwijzingen bevatten naar het document zelf

            Geef alleen de vraag terug, niets anders.""",
            }
        ]

        query = self.llm_service.generate_response(
            user_message=messages[0]["content"]
        ).strip()

        return query

    def verify_triple(self, query: str, pos_text: str, neg_text: str) -> bool:
        """Verify if a triple is valid using LLM.

        Uses the LLM service to verify that a query-positive-negative triple meets
        the requirements for training data.

        Args:
            query: Generated query
            pos_text: Positive example text
            neg_text: Negative example text

        Returns:
            True if triple meets requirements, False otherwise
        """
        message_content = f"""Verifieer deze retrieval triple:

            Zoekopdracht: {query}
            Positief fragment: {pos_text}
            Negatief fragment: {neg_text}

            Vereisten:
            1. De zoekopdracht moet een echte vraag zijn die eindigt met een vraagteken
            2. Beide fragmenten moeten betekenisvolle tekst bevatten
            3. Het positieve fragment moet duidelijk relevant zijn voor de vraag
            4. Het negatieve fragment moet irrelevant zijn voor de vraag maar thematisch verwant
            5. De vraag mag geen interne documentreferenties bevatten

            Geef alleen JA als aan alle vereisten is voldaan, anders NEE."""

        response = (
            self.llm_service.generate_response(user_message=message_content)
            .strip()
            .upper()
        )
        return response == "JA"

    def generate_training_data(
        self,
        embedder_samples: int = 10000,
        chunk_reranker_samples: int = 10000,
        doc_reranker_samples: int = 5000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all training data using clustered chunks.

        Generates training data for embedders and rerankers by processing chunks
        cluster by cluster to ensure good distribution of examples.

        Args:
            embedder_samples: Number of triplets for embedder training
            chunk_reranker_samples: Number of pairs for chunk reranker
            doc_reranker_samples: Number of pairs for document reranker

        Returns:
            Dictionary containing generated training data for each model type
        """
        logger.info("Starting training data generation")

        training_data: Dict[str, List[Dict[str, Any]]] = {
            "embedder_triplets": [],
            "chunk_reranker_pairs": [],
            "document_reranker_pairs": [],
        }

        # Generate data cluster by cluster to ensure good distribution
        samples_per_cluster = {
            "embedder": embedder_samples // self.num_clusters,
            "chunk_reranker": chunk_reranker_samples // self.num_clusters,
            "doc_reranker": doc_reranker_samples // self.num_clusters,
        }

        for _, cluster_chunks in tqdm(
            self.chunk_clusters.items(), desc="Processing clusters"
        ):
            # Generate embedder triplets
            triplets = self._generate_cluster_triplets(
                cluster_chunks, samples_per_cluster["embedder"]
            )
            training_data["embedder_triplets"].extend(triplets)

            # Generate chunk reranker pairs
            chunk_pairs = self._generate_reranker_pairs(
                cluster_chunks, samples_per_cluster["chunk_reranker"]
            )
            training_data["chunk_reranker_pairs"].extend(chunk_pairs)

            # Generate document-level reranker pairs also from chunks
            doc_pairs = self._generate_reranker_pairs(
                cluster_chunks, samples_per_cluster["doc_reranker"]
            )
            training_data["document_reranker_pairs"].extend(doc_pairs)

        return training_data

    def _cluster_chunks(self) -> Dict[int, List[DocumentChunk]]:
        """Cluster chunks using TF-IDF and k-means.

        Clusters document chunks using TF-IDF vectors and k-means clustering to
        ensure diverse training examples.

        Returns:
            Dictionary mapping cluster IDs to lists of chunks in that cluster
        """
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000)
        chunk_contents = [chunk.content for chunk in self.chunks]
        chunk_vectors = vectorizer.fit_transform(chunk_contents)  # type: ignore

        # Cluster chunks
        kmeans = MiniBatchKMeans(n_clusters=min(self.num_clusters, len(self.chunks)))
        chunk_clusters = kmeans.fit_predict(chunk_vectors)  # type: ignore

        # Group chunks by cluster
        clusters: Dict[int, List[DocumentChunk]] = defaultdict(list)
        for chunk, cluster_id in zip(self.chunks, chunk_clusters):  # type: ignore
            clusters[int(cluster_id)].append(chunk)  # type: ignore

        return dict(clusters)

    def _generate_cluster_triplets(
        self, cluster_chunks: List[DocumentChunk], num_samples: int
    ) -> List[Dict[str, str]]:
        """Generate training triplets from chunks in a cluster.

        Creates query-positive-negative triplets using chunks from a single cluster,
        using BM25 to find hard negatives.

        Args:
            cluster_chunks: List of chunks from one cluster
            num_samples: Number of triplets to generate

        Returns:
            List of dictionaries containing query, positive and negative examples
        """
        triplets: List[Dict[str, str]] = []

        for _ in range(num_samples):
            # Select random chunk as anchor
            anchor_chunk = random.choice(cluster_chunks)

            # Find hard negatives using BM25
            hard_negatives_result = self.document_retriever.retrieve(
                [anchor_chunk.content], k=5
            )
            if not hard_negatives_result or not hard_negatives_result[0]:
                continue

            hard_negatives = hard_negatives_result[0]

            # Filter out the anchor chunk
            hard_negative_chunks = [
                chunk
                for chunk in self.chunks
                if any(
                    neg_content in chunk.content
                    for neg_content in [
                        neg_chunk.content for neg_chunk in hard_negatives
                    ]
                )
                and chunk.uuid != anchor_chunk.uuid
            ]

            if not hard_negative_chunks:
                continue

            # Generate query
            query = self.generate_query(anchor_chunk.content)

            # Try each hard negative
            for neg_chunk in hard_negative_chunks:
                if self.verify_triple(query, anchor_chunk.content, neg_chunk.content):
                    triplets.append(
                        {
                            "query": query,
                            "positive": anchor_chunk.content,
                            "negative": neg_chunk.content,
                        }
                    )
                    break

        return triplets

    def _generate_reranker_pairs(
        self,
        cluster_chunks: List[DocumentChunk],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate training pairs for rerankers from chunks.

        Creates query-chunk pairs with binary relevance labels using chunks from
        a single cluster, using BM25 to find hard negatives.

        Args:
            cluster_chunks: List of chunks from one cluster
            num_samples: Number of pairs to generate

        Returns:
            List of dictionaries containing query, chunk and relevance label
        """
        pairs: List[Dict[str, Any]] = []

        for _ in range(num_samples):
            # Select random chunk
            pos_chunk = random.choice(cluster_chunks)

            # Find hard negatives
            hard_negatives_result = self.document_retriever.retrieve(
                [pos_chunk.content], k=5
            )
            if not hard_negatives_result or not hard_negatives_result[0]:
                continue

            hard_negatives = hard_negatives_result[0]

            # Filter out the positive chunk
            hard_negative_chunks = [
                chunk
                for chunk in self.chunks
                if any(
                    neg_content in chunk.content
                    for neg_content in [
                        neg_chunk.content for neg_chunk in hard_negatives
                    ]
                )
                and chunk.uuid != pos_chunk.uuid
            ]

            if not hard_negative_chunks:
                continue

            # Generate query
            query = self.generate_query(pos_chunk.content)

            # Add positive pair
            pairs.append({"query": query, "text": pos_chunk.content, "label": 1})

            # Add negative pair
            neg_chunk = random.choice(hard_negative_chunks)
            pairs.append({"query": query, "text": neg_chunk.content, "label": 0})

        return pairs

    def save_training_data(
        self, data: Dict[str, List[Dict[str, Any]]], output_path: str
    ) -> None:
        """Save generated training data to file.

        Args:
            data: Dictionary containing generated training data
            output_path: Path to save the training data JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Training data saved to {output_path}")
