import { ContextDocuments } from './context';

export interface DocumentChunk {
  uuid: number;
  type?: string;
  link: string;
  content: string;
  parent_document?: string;
  content_date?: string;
  subject?: string;
  first_mentioned_date?: string;
  last_mentioned_date?: string;
  index?: number;
}

export interface ChunkedDocument {
  uuid: string;
  vws_id: string;
  create_date: string;
  type?: string;
  link: string;
  attachment_links: string[];
  content: string;
  chunks: DocumentChunk[];
  subject?: string;
  first_mentioned_date?: string;
  last_mentioned_date?: string;
}

export interface StandardChatResponse {
  response: string;
  chunk_ids: string[];
  document_ids: string[];
  timestamp?: string;
  reasoning_steps?: string[];
}

export interface RetrievalIDsResponse {
  chunk_ids: string[];
  document_ids: string[];
  response?: string;
  timestamp?: string;
  reasoning_steps?: string[];
}

export interface ChunkBatchResponse {
  chunks: DocumentChunk[];
}

export interface DocumentBatchResponse {
  documents: ChunkedDocument[];
}

export interface AdaptiveReasoningUpdate {
  type: 'reasoning_update';
  reasoning_steps: string[];
}

export interface AdaptiveFinalResponse {
  type: 'final_response';
  response: string;
  chunk_ids: string[];
  document_ids: string[];
  timestamp?: string;
  reasoning_steps: string[];
}

export interface AdaptiveErrorResponse {
  type: 'error';
  error: string;
  details?: any;
}

export type AdaptiveSSEResponse = AdaptiveReasoningUpdate | AdaptiveFinalResponse | AdaptiveErrorResponse;

export interface ErrorResponse {
  error: string;
  details?: string;
  status_code?: number;
} 