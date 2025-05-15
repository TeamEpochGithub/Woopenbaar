# Raggle Frontend

The Raggle frontend is a Next.js application that provides a user interface for interacting with the Raggle RAG system. It features a chat interface, document viewing, and advanced filters for retrieving contextually relevant information.

## Tech Stack

- **Framework**: Next.js 14
- **UI**: React 18 with TypeScript
- **Styling**: TailwindCSS
- **Markdown**: Marked for rendering markdown content
- **Event Streaming**: EventSource parser for SSE handling

## Directory Structure

```
frontend/
├── public/                # Static assets
├── src/                   # Source code
│   ├── components/        # UI components
│   │   ├── adaptive/      # Components for adaptive RAG
│   │   ├── chat/          # Chat interface components
│   │   ├── common/        # Shared UI components
│   │   └── filters/       # Filter UI components
│   ├── lib/               # Utility libraries
│   ├── pages/             # Next.js pages
│   ├── services/          # API services
│   ├── styles/            # Global styles
│   └── types/             # TypeScript type definitions
├── tailwind.config.js     # Tailwind configuration
├── next.config.js         # Next.js configuration
└── package.json           # Dependencies and scripts
```

## Key Features

### Chat Interface

- Standard and adaptive RAG chat modes
- Realtime streaming of reasoning steps
- Markdown rendering for formatted responses
- Document reference display

### Document Visualization

- View source documents and chunks
- Highlight relevant passages
- View document metadata

### Filtering

- Filter by document type
- Filter by time period
- Apply custom filter criteria

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Raggle backend running (see backend README)

### Installation

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# or with yarn
yarn install
```

### Configuration

Create a `.env.local` file in the frontend directory with the following variables:

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Running the Development Server

```bash
# Start the development server
npm run dev

# or with yarn
yarn dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

### Building for Production

```bash
# Build the application
npm run build

# or with yarn
yarn build

# Start the production server
npm run start

# or with yarn
yarn start
```

## API Integration

The frontend communicates with the backend through the API client (`src/lib/api.ts`), which provides methods for:

- Sending standard chat messages
- Streaming adaptive chat responses
- Retrieving document chunks
- Fetching document metadata

## Available Pages

- **/** - Main chat interface
- **/sources** - Document sources explorer
- **/explanation** - System explanation page

## Development Guide

### Adding a New Component

1. Create component file in appropriate directory under `src/components/`
2. Define TypeScript props interface
3. Implement the component with proper typing
4. Export the component

### Adding a New Page

1. Create page file in `src/pages/`
2. Import required components
3. Implement page layout and functionality
4. Add any required API calls using the API client

### Styling

The application uses TailwindCSS for styling. Custom styles can be added in:

- `src/styles/globals.css` for global styles
- Component-specific module CSS files for scoped styles 