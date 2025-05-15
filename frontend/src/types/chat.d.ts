import { ContextDocuments } from './context';
import { DocumentChunk, ChunkedDocument } from './api';

export interface ChatMessageData {
  message: string;
  isUser: boolean;
  reasoning?: string[];
  chunks?: DocumentChunk[];
  documents?: ChunkedDocument[];
  chunk_ids?: string[];
  document_ids?: string[];
  timestamp?: string;
} 