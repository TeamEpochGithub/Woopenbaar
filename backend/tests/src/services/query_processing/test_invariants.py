"""Test script to validate invariants in the query processing service.

This script takes a JSON file with queries as input, initializes the query processing
service using the factory method, and runs both the adaptive and standard query
processing on all queries to verify that all invariants (asserts) hold.

Example usage:
    python -m backend.tests.src.services.query_processing.test_invariants \
        --json-path data/train/chunk_query_pairs.json \
        --limit 10
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    cast,
    runtime_checkable,
)

from tqdm import tqdm

# Add the project root to sys.path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from backend.src.data_classes.chunked_document import ChunkedDocument
from backend.src.data_classes.document_chunk import DocumentChunk
from backend.src.services.factory import (
    create_llm_service,
    create_query_processing_service,
    create_retrieval_service,
    create_safety_service,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Protocols for duck typing
@runtime_checkable
class DictConvertible(Protocol):
    """Protocol for objects that can be converted to dict."""

    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class JSONConvertible(Protocol):
    """Protocol for objects that can be converted to JSON-compatible dict."""

    def to_json(self) -> Dict[str, Any]: ...


# Type definitions for JSON data
class QueryItem(TypedDict):
    """Type definition for a query item in the JSON data."""

    query: str
    chunk: str


@dataclass
class QueryProcessingResult:
    """Detailed result of a single query processing operation."""

    query: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    iterations: int = 0


@dataclass
class DetailedResults:
    """Detailed test results for all queries."""

    standard: List[QueryProcessingResult] = field(default_factory=list)
    adaptive: List[QueryProcessingResult] = field(default_factory=list)


# Type definition for summary test results
class CategoryResults(TypedDict):
    """Type definition for test results by category."""

    success: List[str]
    failed: List[str]


class TestResults(TypedDict):
    """Type definition for overall test results."""

    standard: CategoryResults
    adaptive: CategoryResults


class InvariantTest:
    """Test all invariants in the query processing service with actual queries."""

    def __init__(self) -> None:
        """Initialize the invariant test with required services."""
        # Initialize services
        logger.info("Initializing services from cache...")
        self.llm_service = create_llm_service()
        self.retrieval_service = create_retrieval_service()
        self.safety_service = create_safety_service()
        self.query_processing_service = create_query_processing_service(
            llm_service=self.llm_service,
            retrieval_service=self.retrieval_service,
            safety_service=self.safety_service,
        )
        logger.info("Services initialized successfully")

    def load_queries(self, json_path: str, limit: Optional[int] = None) -> List[str]:
        """Load queries from a JSON file.

        Args:
            json_path: Path to the JSON file containing queries
            limit: Optional limit on the number of queries to load

        Returns:
            List of query strings
        """
        logger.info(f"Loading queries from {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data)}")

            queries: List[str] = []
            for item_data in data:
                # Type and validate each item
                if not isinstance(item_data, dict) or "query" not in item_data:
                    logger.warning(
                        f"Skipping invalid item without 'query' field: {item_data}"
                    )
                    continue
                # Cast to proper type
                typed_item: QueryItem = cast(QueryItem, item_data)
                queries.append(typed_item["query"])

            if limit is not None and limit > 0:
                queries = queries[:limit]

            logger.info(f"Loaded {len(queries)} queries")
            return queries

        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            return []

    def test_standard_query(self, query: str) -> QueryProcessingResult:
        """Test a standard query and verify all invariants hold.

        Args:
            query: The query to process

        Returns:
            QueryProcessingResult with processing details
        """
        result = QueryProcessingResult(query=query, success=False)

        try:
            query_result = self.query_processing_service.process_standard_query(
                query=query
            )
            result.success = True

            # Convert QueryResult to dict safely
            if query_result:
                if isinstance(query_result, DictConvertible):
                    result.result = query_result.to_dict()
                else:
                    # Fallback: Convert to dict manually if to_dict() is not available
                    result.result = self._object_to_dict(query_result)

                # Try to extract iterations count
                if hasattr(query_result, "iterations"):
                    result.iterations = getattr(query_result, "iterations")

            return result
        except AssertionError as e:
            result.error_type = "AssertionError"
            result.error_message = str(e)
            logger.error(f"Assertion error in standard query processing: {e}")
            logger.error(f"Query: '{query}'")
            return result
        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.error(f"Error in standard query processing: {e}")
            logger.error(f"Query: '{query}'")
            return result

    def test_adaptive_query(self, query: str) -> QueryProcessingResult:
        """Test an adaptive query and verify all invariants hold.

        Args:
            query: The query to process

        Returns:
            QueryProcessingResult with processing details
        """
        result = QueryProcessingResult(query=query, success=False)

        try:
            query_result = self.query_processing_service.process_adaptive_query(
                query=query
            )
            result.success = True

            # Convert QueryResult to dict safely
            if query_result:
                if isinstance(query_result, DictConvertible):
                    result.result = query_result.to_dict()
                else:
                    # Fallback: Convert to dict manually if to_dict() is not available
                    result.result = self._object_to_dict(query_result)

                # Try to extract iterations count
                if hasattr(query_result, "iterations"):
                    result.iterations = getattr(query_result, "iterations")

            return result
        except AssertionError as e:
            result.error_type = "AssertionError"
            result.error_message = str(e)
            logger.error(f"Assertion error in adaptive query processing: {e}")
            logger.error(f"Query: '{query}'")
            return result
        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.error(f"Error in adaptive query processing: {e}")
            logger.error(f"Query: '{query}'")
            return result

    def _object_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert an object to a dictionary by extracting its attributes.

        Args:
            obj: The object to convert

        Returns:
            Dictionary representation of the object
        """
        # Handle DocumentChunk and ChunkedDocument objects
        if isinstance(obj, (DocumentChunk, ChunkedDocument)):
            if hasattr(obj, "to_json") and callable(getattr(obj, "to_json")):
                return obj.to_json()

        # Handle lists of objects
        if isinstance(obj, list):
            return cast(
                Dict[str, Any],
                {"_list_items": [self._object_to_dict(item) for item in obj]},
            )

        # Handle dictionaries
        if isinstance(obj, dict):
            result_dict: Dict[str, Any] = {}
            for k, v in obj.items():
                result_dict[str(k)] = self._object_to_dict(v)
            return result_dict

        # If the object has __dict__, use it to extract attributes
        if hasattr(obj, "__dict__"):
            result: Dict[str, Any] = {}
            for key, value in obj.__dict__.items():
                # Skip private attributes
                if key.startswith("_"):
                    continue
                result[key] = self._object_to_dict(value)
            return result

        # Fall back to the object itself if it's a primitive type
        # This should be a JSON serializable type or the serialization will fail
        if isinstance(obj, (str, int, float, bool, type(None))):
            return cast(
                Dict[str, Any], obj
            )  # Type cast to satisfy linter, we know this isn't a Dict

        # For other types, convert to string
        return {"_unserializable": str(obj)}

    def run_tests(
        self, queries: List[str], randomize: bool = True
    ) -> Tuple[TestResults, DetailedResults]:
        """Run both standard and adaptive tests on all queries.

        Args:
            queries: List of queries to test
            randomize: Whether to randomize the order of queries

        Returns:
            Tuple of (summary results, detailed results)
        """
        summary_results: TestResults = {
            "standard": {"success": [], "failed": []},
            "adaptive": {"success": [], "failed": []},
        }

        detailed_results = DetailedResults()

        # Create a copy of queries to process
        processing_queries = queries.copy()

        # Randomize order if requested
        if randomize:
            random.shuffle(processing_queries)

        # Show progress bar
        pbar = tqdm(total=len(processing_queries) * 2, desc="Processing queries")

        for query in processing_queries:
            # Test standard query
            standard_result = self.test_standard_query(query)
            detailed_results.standard.append(standard_result)

            if standard_result.success:
                summary_results["standard"]["success"].append(query)
            else:
                summary_results["standard"]["failed"].append(query)

            pbar.update(1)

            # Test adaptive query
            adaptive_result = self.test_adaptive_query(query)
            detailed_results.adaptive.append(adaptive_result)

            if adaptive_result.success:
                summary_results["adaptive"]["success"].append(query)
            else:
                summary_results["adaptive"]["failed"].append(query)

            pbar.update(1)

        pbar.close()
        return summary_results, detailed_results


def main() -> None:
    """Parse arguments and run invariant tests."""
    parser = argparse.ArgumentParser(
        description="Test query processing service invariants with real queries"
    )
    parser.add_argument(
        "--json-path", type=str, required=True, help="Path to JSON file with queries"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of queries to test (default: test all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file for results (default: print to console)",
    )
    parser.add_argument(
        "--detailed-output",
        type=str,
        default=None,
        help="Path to output detailed JSON results",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Process queries in order instead of randomly",
    )

    args = parser.parse_args()

    # Initialize test
    test = InvariantTest()

    # Load queries
    queries = test.load_queries(args.json_path, args.limit)
    if not queries:
        logger.error("No queries loaded, exiting")
        sys.exit(1)

    # Run tests
    logger.info(f"Running tests on {len(queries)} queries")
    summary_results, detailed_results = test.run_tests(
        queries, randomize=not args.no_randomize
    )

    # Print summary
    total_queries = len(queries)
    standard_success = len(summary_results["standard"]["success"])
    adaptive_success = len(summary_results["adaptive"]["success"])

    logger.info(f"Tests completed: {total_queries} queries tested")
    logger.info(
        f"Standard query success rate: {standard_success}/{total_queries} ({standard_success/total_queries*100:.1f}%)"
    )
    logger.info(
        f"Adaptive query success rate: {adaptive_success}/{total_queries} ({adaptive_success/total_queries*100:.1f}%)"
    )

    # Output results if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary results saved to {args.output}")

    # Output detailed results if requested
    if args.detailed_output:
        # Convert dataclass instances to dictionaries for JSON serialization
        detailed_dict = {
            "standard": [asdict(r) for r in detailed_results.standard],
            "adaptive": [asdict(r) for r in detailed_results.adaptive],
        }

        with open(args.detailed_output, "w", encoding="utf-8") as f:
            json.dump(detailed_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {args.detailed_output}")

    # Exit with error code if any tests failed
    if (
        len(summary_results["standard"]["failed"]) > 0
        or len(summary_results["adaptive"]["failed"]) > 0
    ):
        logger.error("Some tests failed")
        sys.exit(1)
    else:
        logger.info("All tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
