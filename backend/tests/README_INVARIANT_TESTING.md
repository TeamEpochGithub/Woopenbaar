# Invariant Testing for Query Processing Service

This directory contains a script for testing invariants in the Query Processing Service using real queries from a JSON file.

## Purpose

The `test_invariants.py` script validates that all invariants (assertions) in the query processing pipeline hold true when processing real queries. This helps ensure that the relationships between state, reasoning steps, and frontend updates remain consistent across different types of queries.

## Prerequisites

- A valid RAG cache (the retrieval service is loaded from cache)
- A JSON file containing queries in the following format:

```json
[
  {
    "query": "What is the policy on remote work?",
    "chunk": "Employees can work remotely up to 3 days per week..."
  },
  {
    "query": "Another query",
    "chunk": "Another chunk..."
  }
]
```

## Usage

Run the script with the following command:

```bash
python -m backend.tests.src.services.query_processing.test_invariants \
    --json-path data/train/chunk_query_pairs.json \
    --limit 10 \
    --output results.json
```

### Arguments

- `--json-path`: (Required) Path to the JSON file containing queries
- `--limit`: (Optional) Limit the number of queries to test (default: test all)
- `--output`: (Optional) Path to save the test results as JSON (default: print to console only)

## Output

The script will:

1. Load queries from the specified JSON file
2. Initialize the query processing service from cache
3. Run both `process_standard_query` and `process_adaptive_query` on each query
4. Check if any invariants (assertions) fail
5. Generate a summary of success and failure rates
6. (Optional) Save detailed results to a JSON file

### Sample Output

```
2023-05-15 10:25:34 - __main__ - INFO - Tests completed: 10 queries tested
2023-05-15 10:25:34 - __main__ - INFO - Standard query success rate: 10/10 (100.0%)
2023-05-15 10:25:34 - __main__ - INFO - Adaptive query success rate: 9/10 (90.0%)
2023-05-15 10:25:34 - __main__ - ERROR - Some tests failed
```

## Exit Codes

- `0`: All tests passed (all invariants hold for all queries)
- `1`: Some tests failed (at least one invariant was violated)

## Troubleshooting

If you encounter any issues:

1. Check that the JSON file exists and has the correct format
2. Verify that the RAG cache exists and is properly initialized
3. Look for specific assertion error messages in the logs
4. Use a smaller `--limit` value to narrow down problematic queries 