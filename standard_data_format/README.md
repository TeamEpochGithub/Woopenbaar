# Standard Data Format Processor

A distributed document processing system that converts PDF documents into a standardized markdown format using a tiered approach (regular conversion → forced OCR → LLM), it is also possible to disable options. The system makes use of the marker package (https://github.com/VikParuchuri/marker) for the conversion. The system supports parallel processing across multiple machines.


## System Requirements

- Python 3.8+
- CUDA-capable device for OCR and local LLM
- Downloaded documents PLUS metadata in the data/ directory
- (Optional) SSH access (for distributed mode)
- (Optional) Google API key (for Gemini LLM)
- (Optional) Ollama (for local LLM)


## Quick Start multiprocessing

Before startinging check the capabilities of your gpus each chunk takes about 5 GB of GPU memory, and for local llm inference it is dependent on the size of the llm. 
At the top of the bash scripts the relevant variables are mentioned.

1. Divide metadata into chunks:
```bash
./standard_data_format/scripts/divide_metadata.sh
```

This bash script divides the metadata into a total number of chunks so it can be multiprocessed. 
The variables: 
- METADATA_PATH: local path to the metadata csv file.
- OUTPUT_DIR: output directory for the chunks.
- TOTAL_CHUNKS: total amount of chunks, dependant on your gpu's memory.

2.  (Optional) Create worker scripts (variables: Total amount of chunks, number of workers/machines, chunks per worker, source script):
```bash
./standard_data_format/scripts/create_run_workers.sh
```
This bash script copies the example bash scripts and changes the amount of chunks handled by each worker/machine. 
The variables:
- TOTAL_CHUNKS: total amount of chunks, dependant on your gpu's memory.
- NUM_WORKERS: the amount of workers/machines available.
- CHUNKS_PER_WORKER: TOTAL_CHUNKS divided by NUM_WORKERS if the gpus have similar memory capabilities.
- SOURCE_SCRIPT: Which worker script gets copied and rewritten, either run_gemini_worker.sh for online llm use or run_local_worker.sh


3. Run the script on the base machine choose , first take a look at the config files to see what can be adapted:
```bash
./standard_data_format/scripts/run_gemini_worker.sh
# or
./standard_data_format/scripts/run_local_worker.sh
```
The bash scripts for launching multiprocess.py to create standard data format from the created metadata chunks using the gemini llm or ollama llm (local).
The variables for run_gemini_worker:
- BASE_MACHINE: Base machine where the metadata and documents are stored.
- TOTAL_CHUNKS: total amount of chunks, dependant on your gpu's memory.
- START_CHUNK: Start Chunk ID number for this worker, first chunk processed in the range (START_CHUNK to END_CHUNK).
- END_CHUNK: End Chunk ID number for this worker.
- GPU_COUNT: Number of gpus on this machine to use, it will distribute the chunks evenly.
- YAML_CONFIG: path to chosen config for conversion.
- CUSTOM_METADATA_PATH: path to the metadata chunks.

The variables for run_local_worker:
- BASE_MACHINE: SSH connection string for the base machine where metadata and documents are stored
- TOTAL_CHUNKS: Total number of processing chunks, dependent on your GPU's memory
- START_CHUNK: Start Chunk ID number for this worker, first chunk processed in the range
- END_CHUNK: End Chunk ID number for this worker
- OLLAMA_GPU: GPU ID dedicated to running the Ollama LLM service (typically 0)
- PROCESSING_GPU: GPU ID used for document processing (typically 1)
- OLLAMA_STARTUP_WAIT: Number of seconds to wait for Ollama service to start
- LOCK_DIR: Directory path for storing process lock files
- OLLAMA_LOG: Path to the Ollama service log file
- SYNC_INTERVAL: How often to sync processed documents with base machine (in number of documents)
- YAML_CONFIG: path to chosen config for conversion.
4. (Optional) Run the script on the seperate worker machines:
```bash
./standard_data_format/scripts/run_worker_1.sh
./standard_data_format/scripts/run_worker_2.sh
./standard_data_format/scripts/run_worker_3.sh
...
```

## Architecture

### Core Components in src

1. **Document Processing Pipeline**
   - `multiprocess.py`: Entry point and orchestrator
   - `document_processing_pipeline.py`: End-to-end workflow manager
   - `pdf_converter.py`: Tiered PDF conversion (regular → OCR → LLM)
   - `document_processor.py`: Document and attachment processor

2. **Data Management**
   - `metadata.py`: Metadata and relationship manager
   - `document.py`: Document format definition
   - `divide_metadata.py`: Metadata distribution tool


### Python files description

## 1. multiprocess.py - Entry Point and Orchestrator
- Serves as the main entry point for the document processing system
- Handles command-line arguments for distributed processing configuration
- Loads and merges configuration from YAML files
- Configures device allocation (CUDA) for processing
- Sets up distributed processing with chunk assignment
- Manages logging configuration for each worker process
- Coordinates communication between worker machines

## 2. document_processing_pipeline.py - Workflow Manager
- Controls the end-to-end document processing workflow
- Validates and applies configuration settings
- Creates necessary directory structures
- Initializes and coordinates component services
- Processes document families in parallel with thread-safety
- Implements synchronization between machines
- Saves processed documents and updates metadata
- Provides robust error handling and recovery mechanisms
- Manages file locking to prevent concurrent access issues

## 3. pdf_converter.py - Tiered PDF Conversion
- Implements a fallback strategy for PDF conversion:
  - Regular conversion (text extraction)
  - Forced OCR conversion (for scanned documents)
  - LLM-based conversion (for complex documents)
- Manages GPU resources for processing
- Supports both local (Ollama) and cloud (Gemini) LLM services
- Configures the Marker library for optimal performance
- Provides quality assessment of converted content
- Implements thread-safe buffer management
- Handles Dutch-specific document formatting

## 4. document_processor.py - Document and Attachment Processor
- Handles document relationships and family hierarchies
- Processes base documents and their attachments
- Combines related content with proper formatting
- Generates deterministic UUIDs for document tracking
- Implements content caching for performance
- Ensures thread-safe operations with locks
- Handles collisions and content deduplication
- Manages document metadata and relationships

## 5. metadata.py - Metadata and Relationship Manager
- Loads and validates document metadata from CSV files
- Tracks document families and attachments
- Manages document processing status updates
- Identifies document availability and location
- Normalizes document IDs for consistent processing
- Adds UUID and attachment relationship information
- Creates family mappings for efficient processing
- Implements thread-safe metadata access and updates
- Provides document lookup by various identifiers

## 6. document.py - Standard Document Format Definition
- Defines the core document data structure
- Implements JSON serialization and deserialization
- Provides factory methods for document creation
- Maintains standardized document metadata format
- Handles document content formatting
- Supports attachment linking and organization

## 7. divide_metadata.py - Metadata Distribution Tool
- Divides metadata into balanced chunks for distributed processing
- Preserves document family relationships when splitting
- Supports family-based or random distribution strategies
- Creates separate metadata files for each processing chunk
- Provides command-line interface for configuration

## 8. custom_marker.py - Marker Library Customization
- Customizes the Marker library for Dutch document processing
- Implements specialized image description for government documents
- Enhances layout processing for Dutch official formats
- Provides Dutch-specific prompts for LLM conversion
- Handles sensitive information and redactions
- Optimizes batch processing and resource utilization

### Processing Flow

1. **Metadata Division**
   - Groups documents by family
   - Divides into balanced chunks
   - Preserves relationships

2. **Document Processing**
   - Attempts regular conversion
   - Falls back to OCR if needed
   - Uses LLM as final option
   - Combines attachments with main documents

3. **Synchronization**
   - Saves processed documents
   - Updates metadata status
   - Syncs with base machine

## Input Requirements

### Metadata CSV Structure
```csv
ID,Matter,Family,Document,File Type,Datum,Document Link,besluit_id,available
```

Key Fields:
- `ID`: Unique identifier (e.g., "Matter-123")
- `Family`: Groups related documents
- `Document`: Title/name for attachment matching
- `besluit_id`: Decision reference (falls back to Matter)

### Document Organization
- PDFs in `data/timelines/all_documents_id/`
- Naming patterns:
  - `{doc_id}.pdf`
  - `{besluit_id}_{doc_id}.pdf`
  - `{matter}_{doc_id}.pdf`

## Output Format

```json
{
    "uuid": "deterministic-uuid",
    "vws_id": "original-doc-id",
    "create_date": "YYYY-MM-DD",
    "type": "document-type",
    "link": "original-url",
    "attachment_links": ["url1", "url2"],
    "content": "markdown-content"
}
```

### Content Structure
```markdown
[Main Document Content]

Begin Attachment: {id}, {name}, {link}
[Attachment Content]
End Attachment: {id}, {name}, {link}
```

## Configuration

### Key Settings (gemini_config.yaml or local_config.yaml)
```yaml
marker:
  output_format: "markdown"
  use_local_model: false
  llm_service: "marker.services.gemini.GoogleGeminiService" 
  batch_size: 1

pdf_converter:
  allow_regular_conversion: false
  allow_ocr: false
  allow_llm: true
```

## Error Handling & Performance

- Tiered conversion fallback
- Thread-safe operations
- Process and file locking
- Progress tracking
- Regular synchronization
- Load balancing
- Detailed logging

## Directory Structure
```

Raggle
├── standard_data_format/
│   ├── output_scraped/
│   │   ├──json_files/
│   │   └── timelines/ 
│   ├── scripts/
│   │   ├── create_run_workers.sh
│   │   ├── distribute_metadata.sh
│   │   ├── divide_metadata.sh
│   │   ├── run_gemini_worker.sh
│   │   ├── run_local_worker.sh
│   │   └── run_worker_*.sh
│   ├── src/
│   │   └── multiprocess.py
│   ├── config/
│   │   ├── gemini_config.yaml
│   │   └── local_config.yaml
│   └── requirements.txt
├── data/
│   ├── metadata/
│   │   ├── divided/
│   │   └── all.csv
│   └── woo_scraped/
│       └── documents/
│       └── timelines/
└── .venv/
```

## Logging

- Detailed logs in `logs` directory
- Per-chunk log files
- Configurable verbosity
- Error tracking and reporting

## Custom Marker Modifications

The project includes several customizations to the Marker library's default behavior through `custom_marker.py`:

1. **Dutch Language Support**: Some prompts have been translated and optimized for Dutch language processing:
   - Image description prompt (`custom_image_description_prompt`)
   - Layout relabeling prompts (`custom_topk_relabelling_prompt` and `custom_complex_relabeling_prompt`)
   - Complex region formatting prompt (`custom_complex_region_prompt`)

2. **Enhanced Image Description**: The image description processor has been modified to:
   - Handle Dutch government documentation specifically
   - Process redacted text marked with "5.1.2e" references
   - Generate descriptions in Dutch with 3-4 sentences
   - Include specific document context and visual elements

3. **Layout Processing**: Custom implementations for:
   - `surya_layout` method with configurable batch processing
   - Layout model configuration with tqdm progress bar control
   - High-resolution image processing support

4. **Prompt Customizations**: All prompts have been tailored for:
   - Dutch government document context
   - Specific formatting requirements
   - Handling of sensitive information
   - Structured output formatting

These modifications are applied by overriding the default Marker library methods:
```python
LLMLayoutBuilder.surya_layout = custom_surya_layout
LLMLayoutBuilder.topk_relabelling_prompt = custom_topk_relabelling_prompt
LLMLayoutBuilder.complex_relabeling_prompt = custom_complex_relabeling_prompt
LLMImageDescriptionProcessor.image_description_prompt = custom_image_description_prompt
LLMComplexRegionProcessor.complex_region_prompt = custom_complex_region_prompt
```
