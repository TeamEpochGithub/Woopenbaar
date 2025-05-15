# Store Service

The Store Service provides persistent storage functionality for the Raggle backend, currently focusing on chat history storage and retrieval using JSON files.

## Features

- **Chat History Storage**: Persists conversation history to JSON files
- **Thread-Safe Operations**: Handles concurrent access with thread locks
- **Pagination Support**: Retrieves history with limit and offset capabilities
- **Automatic Initialization**: Creates storage files if they don't exist

## Components

### ChatHistoryService

The main service class that provides storage and retrieval capabilities:

- `store_chat_history`: Saves a ChatHistoryRecord to persistent storage
- `store_chat_with_chunks`: Helper method that takes individual parameters and creates a ChatHistoryRecord
- `get_chat_history`: Retrieves chat history with pagination support
- `initialize_storage`: Creates the necessary storage files if they don't exist

## Implementation Details

- Uses JSON files for simple, human-readable storage
- Implements thread locks to ensure safe concurrent access
- Sorts chat history by timestamp for consistent retrieval order
- Handles error conditions gracefully with logging

## ChatHistoryRecord Structure

The `ChatHistoryRecord` class is used to store chat history items with the following fields:

```python
@dataclass
class ChatHistoryRecord:
    timestamp: str
    question: str
    response: str
    document_chunks: List[DocumentChunk] = field(default_factory=list)
    data_sources: Set[str] = field(default_factory=set)
    chat_type: str = "standard_rag"
```

Each chat history entry is stored as a JSON object with these fields:

```json
{
  "timestamp": "2023-06-01 14:30:45",
  "question": "What is the vacation policy?",
  "response": "Employees are entitled to 20 days of vacation per year...",
  "document_chunks": [{"chunk_content": "...", "metadata": {...}}],
  "data_sources": ["source_1", "source_2"],
  "chat_type": "standard_rag"
}
```

## Configuration

Key configuration parameter:

- `CHAT_HISTORY_PATH`: Path to the JSON file for chat history storage

## File Structure

The chat history is stored as a JSON array where each item represents a single chat interaction:

```json
[
  {
    "timestamp": "2023-06-01 15:30:45",
    "question": "What is the vacation policy?",
    "response": "...",
    "document_chunks": [{...}],
    "data_sources": ["..."],
    "chat_type": "standard_rag"
  },
  {
    "timestamp": "2023-06-01 14:30:45",
    "question": "How do I submit expenses?",
    "response": "...",
    "document_chunks": [{...}],
    "data_sources": ["..."],
    "chat_type": "standard_rag"
  }
]
```

## Usage

```python
from backend.src.services.store import ChatHistoryService
from backend.src.data_classes.document_chunk import DocumentChunk
from datetime import datetime

# Initialize the service
history_service = ChatHistoryService()

# Store a chat interaction using the helper method
document_chunks = [DocumentChunk(...), DocumentChunk(...)]
success = history_service.store_chat_with_chunks(
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    question="What is the vacation policy?", 
    response="Employees are entitled to...",
    document_chunks=document_chunks,
    chat_type="standard_rag",
    data_sources=["source1", "source2"]
)

# Retrieve chat history (most recent first)
history = history_service.get_chat_history(limit=10, skip=0)
```

## Thread Safety

All operations are protected by a thread lock to ensure that concurrent accesses don't corrupt the JSON file:

```python
with self.lock:
    # File operations
    # ...
```

## Future Extensions

The service is designed to be extended with additional storage capabilities:

1. Database storage for production use
2. Document metadata storage
3. User session management
4. Analytics data storage 