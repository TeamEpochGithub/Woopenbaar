# Query Processing Service

The Query Processing Service orchestrates the complete query processing flow in the Raggle backend, managing the transformation of user queries into well-structured answers using retrieved document context.

## Features

- **Standard and Adaptive RAG**: Supports both direct and multi-step retrieval approaches
- **Query Analysis**: Evaluates query complexity and determines processing strategy
- **Retrieval Management**: Coordinates document retrieval operations
- **Relevance Evaluation**: Assesses and filters retrieved content for relevance
- **Answer Synthesis**: Generates coherent answers from gathered context
- **Follow-up Questions**: Suggests relevant follow-up questions
- **Progress Tracking**: Maintains state throughout the query processing flow

## Components

### QueryProcessingService

The main service class that orchestrates the complete query pipeline:

- `process_standard_query`: Direct query processing without subqueries
- `process_adaptive_query`: Multi-step query processing with reasoning

### Pipeline Components

The service integrates specialized components that handle different aspects of the pipeline:

- **ProgressEvaluator**: Analyzes query progress and generates subqueries
- **RetrievalManager**: Coordinates document retrieval operations
- **ChunkEvaluator**: Evaluates retrieved chunks for relevance
- **AnswerSynthesizer**: Generates final and intermediate answers
- **SuggestionGenerator**: Creates contextually relevant follow-up questions

## Query Processing Pipelines

### Standard Processing

The standard approach follows a direct path:

1. **Retrieval**: Find chunks relevant to the user's query
2. **Evaluation**: Assess and filter chunks for relevance
3. **Answer Synthesis**: Generate a coherent answer
4. **Follow-up Questions**: Generate suggestions for related queries

### Adaptive Processing

The adaptive approach implements a more complex flow:

1. **Initial Retrieval**: Get initial context for the query
2. **Progress Evaluation**: Determine if more information is needed
3. **Subquery Generation**: Create targeted follow-up questions
4. **Iterative Retrieval**: Retrieve additional context with subqueries
5. **Reasoning Chain**: Build a logical progression of information
6. **Final Answer**: Synthesize a comprehensive response
7. **Follow-up Questions**: Generate contextually relevant suggestions

## Query State Management

The `QueryState` class maintains the state of a query throughout processing:

- Tracks the original query and generated subqueries
- Maintains a context list with all relevant information
- Records the reasoning trace for explainability
- Manages the readiness state for answer generation

## Usage

```python
from backend.src.services.query_processing import QueryProcessingService

# Initialize dependencies
llm_service = create_llm_service()
retrieval_service = create_retrieval_service()
safety_service = create_safety_service()

# Create the service
query_processor = QueryProcessingService(
    llm_service=llm_service,
    retrieval_service=retrieval_service,
    safety_service=safety_service
)

# Process a standard query
result = query_processor.process_standard_query(
    query="What is the vacation policy?",
    filters={"document_type": "Policy"}
)

# Process an adaptive query with progressive retrieval
result = query_processor.process_adaptive_query(
    query="Compare the vacation policies across departments",
    max_iterations=3,
    filters={"document_type": "Policy"}
)

# Access query results
print(f"Query: {result.query}")
print(f"Response: {result.response}")
print(f"Chunks used: {len(result.chunks)}")
print(f"Reasoning steps: {result.reasoning_steps}")
```

## Integration with Other Services

The Query Processing Service integrates with:

- **LLM Service**: For text generation and reasoning
- **Retrieval Service**: For document and chunk retrieval
- **Safety Service**: For content validation 