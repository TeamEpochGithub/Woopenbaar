# Woopenbaar

Woopenbaar is a comprehensive Retrieval-Augmented Generation (RAG) system that combines document processing, semantic search, and large language models to provide context-aware responses to user queries. The system features both standard and adaptive RAG approaches with a modern web interface.

## Components

### Core System

- **Backend**: Python application with Flask API, retrieval engine, and LLM integration
- **Frontend**: Next.js web interface with chat and document visualization

### Data Processing Components

- **Scraper**: Web scraping tools for collecting documents from various sources
- **Standard Data Format**: System for converting documents to a standardized markdown format
- **Trained Models**: Pre-trained models for embeddings, chunk reranking, and safety checks

## Features

### Backend

- **Document Processing**: Index and chunk documents for efficient retrieval
- **Multi-Source Support**: Organize and search across different data sources
- **Semantic Search**: Find content based on meaning, not just keywords
- **Multiple LLM Integrations**: Support for local models via vLLM, Google's Gemini, and DeepSeek
- **Adaptive Reasoning**: Multi-step retrieval with progressive refinement for complex queries
- **Safety Filters**: Content validation and filtering

### Frontend

- **Intuitive Chat Interface**: User-friendly chat UI with conversation history
- **Adaptive Mode**: View reasoning steps in real-time as the system processes complex queries
- **Document Explorer**: View and navigate through source documents and chunks
- **Advanced Filtering**: Filter by document type, time period

## Prerequisites

- **Backend**:
  - Python 3.9+
  - CUDA-compatible GPU(s) if you run LLM locally
  - Python package manager that supports pyproject.toml (pip, uv, poetry, etc.)

- **Frontend**:
  - Node.js 18+
  - npm or yarn

## Installation

### 1. Backend Setup

Create a virtual environment and install dependencies using pyproject.toml:

```bash
# Enter project directory
cd woopenbaar

# Set up virtual environment
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
# Using pip:
pip install -e .

# Or if you prefer uv:
# uv pip install -e .

# Or if you prefer poetry:
# poetry install
```

### 2. Frontend Setup

Install Node.js dependencies:

```bash
cd frontend
npm install
```

## Running the Application

You'll need to run both the backend and frontend servers in separate terminals.

### 1. Start the Backend Server

```bash
cd backend
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python app.py
```

The Flask API server will by default start on http://localhost:5000


### 2. Start the Frontend Server

```bash
cd frontend
npm run dev
```

The Next.js development server will start on http://localhost:3000

## Configuration

### Backend Configuration

Configure the backend through:
- Environment variables for sensitive settings (API keys, etc.)
- Settings in `backend/conf/config.py`
- LLM prompts in `backend/conf/prompts.py'

Key configuration options:
- Language model selection (LOCAL, GEMINI, DEEPSEEK)
- Model parameters (temperature, max tokens, etc.)
- Retrieval parameters (chunk size, document pool size, etc.)

### Frontend Configuration

Create a `.env.local` file in the frontend directory:

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Development

### Backend Development

```bash
cd backend
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python app.py
```

For detailed backend development instructions, see [Backend README](backend/README.md).

### Frontend Development

```bash
cd frontend
npm run dev
```

This starts the development server with hot-reloading enabled.

For detailed frontend development instructions, see [Frontend README](frontend/README.md).

## Project Structure

```
raggle/
├── backend/                # Backend Python application
│   ├── app.py              # Application entry point
│   ├── conf/               # Configuration and prompts
│   ├── pyproject.toml      # Python project metadata and dependencies
│   ├── src/                # Source code
│   │   ├── api/            # API endpoints
│   │   ├── data_classes/   # Data models
│   │   └── services/       # Business logic
│   └── tests/              # Backend tests
├── frontend/               # Frontend Next.js application
│   ├── public/             # Static assets
│   ├── src/                # Source code
│   │   ├── components/     # UI components
│   │   ├── lib/            # Utility libraries
│   │   ├── pages/          # Next.js pages
│   │   ├── services/       # API clients
│   │   └── types/          # TypeScript definitions
│   └── tests/              # Frontend tests
├── scraper/                # Document collection scripts
│   ├── scrape_vws.py       # VWS document scraper
│   ├── scrape_documents.py # General document scraper
│   └── website_specific.py # Website-specific scraping logic
├── standard_data_format/   # Document standardization system
│   ├── config/             # Configuration files
│   ├── scripts/            # Automation scripts
│   ├── src/                # Processing pipeline
│   └── utils/              # Utility functions
├── trained_models/         # Pre-trained ML models
│   ├── embedders/          # Text embedding models
│   ├── chunk_rerankers/    # Document reranker models
│   └── safety/             # Content safety models
└── data/                   # Shared data directory
└── evaluation/             # Evaluation module
```

## Data Collection and Processing

1. **Document Collection** - Use the `scraper` to gather documents:
   ```bash
   cd scraper
   python scrape_vws.py  # For VWS documents
   python scrape_documents.py  # For other document sources
   ```

2. **Document Standardization** - Process collected documents:
   ```bash
   cd standard_data_format
   # First divide metadata into chunks for processing
   ./scripts/divide_metadata.sh
   # Then process with chosen configuration
   ./scripts/run_gemini_worker.sh  # For cloud-based processing
   # or
   ./scripts/run_local_worker.sh  # For local processing
   ```

For more details on data processing, see:
- [Scraper README](scraper/README.md)
- [Standard Data Format README](standard_data_format/README.md)

## Tests

### Backend Tests

```bash
cd backend
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pytest
```

## Contributors

This repository was created by [Team Epoch V](https://teamepoch.ai/team#v), based in
the [Dream Hall](https://www.tudelft.nl/ddream) of the [Delft University of Technology](https://www.tudelft.nl/).



Kenzo Heijman,&nbsp;&nbsp; Marcin Jarosz,&nbsp;&nbsp; Felipe Bononi Bello,&nbsp;&nbsp; Laura Kaczmarzyk,&nbsp;&nbsp; Maxim van Emmerik 

[![Github Badge](https://img.shields.io/badge/-Blagues-24292e?style=flat&logo=Github)](https://github.com/Blagues)
[![Github Badge](https://img.shields.io/badge/-MarJarAI-24292e?style=flat&logo=Github)](https://github.com/MarJarAI)
[![Github Badge](https://img.shields.io/badge/-FBB0-24292e?style=flat&logo=Github)](https://github.com/FBB0)
[![Github Badge](https://img.shields.io/badge/-LauraKaczmarzyk-24292e?style=flat&logo=Github)](https://github.com/LauraKaczmarzyk)
[![Github Badge](https://img.shields.io/badge/-Maxim1920-24292e?style=flat&logo=Github)](https://github.com/Maxim1920)
