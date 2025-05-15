import { FilterOptions } from '@/types/filters';
import { 
  AdaptiveSSEResponse, 
  AdaptiveReasoningUpdate, 
  AdaptiveFinalResponse,
  AdaptiveErrorResponse,
  StandardChatResponse,
  RetrievalIDsResponse,
  ChunkBatchResponse,
  DocumentBatchResponse,
  DocumentChunk,
  ChunkedDocument
} from '@/types/api';
import { AdaptiveSettings } from '@/types/adaptive';
import { createParser, ParseEvent } from 'eventsource-parser';

export interface ApiRequestOptions {
  signal?: AbortSignal;
}

export class ApiClient {
  private baseUrl: string;
  
  constructor(baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  // Common fetch wrapper with error handling
  private async fetch<T>(endpoint: string, options: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Send a regular chat message to the backend
   */
  async sendChatMessage(
    message: string, 
    filters: FilterOptions,
    prioritize_earlier: boolean = false,
    options?: ApiRequestOptions
  ): Promise<StandardChatResponse> {
    console.log('api: Initiating standard chat request', {
      message,
      filters,
      prioritize_earlier,
    });

    const response = await this.fetch<StandardChatResponse>('/chat/with-context', {
      method: 'POST',
      body: JSON.stringify({ message, filters, prioritize_earlier }),
      signal: options?.signal,
    });

    console.log('api: Received standard chat response', {
      hasResponse: Boolean(response.response),
      hasChunkIds: Boolean(response.chunk_ids?.length),
      chunkIdsCount: response.chunk_ids?.length,
    });

    // Initialize empty arrays if not present
    if (!response.chunk_ids) response.chunk_ids = [];
    if (!response.document_ids) response.document_ids = [];

    return response;
  }

  /**
   * Retrieve context without generating a response
   */
  async retrieveContext(
    message: string,
    filters: FilterOptions,
    options?: ApiRequestOptions
  ): Promise<RetrievalIDsResponse> {
    return this.fetch<RetrievalIDsResponse>('/retrieve-context', {
      method: 'POST',
      body: JSON.stringify({ message, filters }),
      signal: options?.signal,
    });
  }

  /**
   * Get a batch of chunks by their IDs
   */
  async getChunksBatch(chunkIds: string[]): Promise<ChunkBatchResponse> {
    // Filter out any empty or invalid IDs
    const validIds = chunkIds
      .filter(id => id && id.trim().length > 0)
      .map(id => id.trim().replace(/[^\w\d-]/g, ''))  // Sanitize IDs
      .filter(id => id.length > 0 && id.length < 30); // Basic length validation
    
    if (validIds.length === 0) {
      console.warn('No valid chunk IDs provided to getChunksBatch');
      return { chunks: [] };
    }
    
    // Log the IDs being sent for debugging
    console.log('Sending chunk IDs to API:', validIds.slice(0, 3), `... (${validIds.length} total)`);
    
    try {
      return this.fetch<ChunkBatchResponse>('/chunks-batch', {
        method: 'POST',
        body: JSON.stringify({ chunk_ids: validIds }),
      });
    } catch (error) {
      console.error('Error fetching chunks batch:', error);
      return { chunks: [] };
    }
  }

  /**
   * Get a batch of documents by their UUIDs
   */
  async getDocumentsBatch(documentIds: string[]): Promise<DocumentBatchResponse> {
    // Filter out any empty or undefined IDs
    const validIds = documentIds.filter(id => id && id.trim().length > 0);
    
    if (validIds.length === 0) {
      console.warn('No valid document IDs provided to getDocumentsBatch');
      return { documents: [] };
    }
    
    try {
      const response = await this.fetch<DocumentBatchResponse>('/documents-batch', {
        method: 'POST',
        body: JSON.stringify({ document_ids: validIds }),
      });
      
      // If documents don't have chunks array, initialize it
      response.documents = response.documents.map(doc => {
        if (!doc.chunks) {
          doc.chunks = [];
        }
        return doc;
      });
      
      return response;
    } catch (error) {
      console.error('Error fetching documents batch:', error);
      return { documents: [] };
    }
  }

  /**
   * Send an adaptive chat message with streaming responses
   */
  async sendAdaptiveChatMessage(
    message: string,
    filters: FilterOptions,
    adaptiveSettings: AdaptiveSettings,
    callbacks: {
      onReasoningUpdate?: (reasoningSteps: string[]) => void;
      onFinalResponse?: (data: AdaptiveFinalResponse) => void;
      onError?: (error: string) => void;
    },
    options?: ApiRequestOptions
  ): Promise<void> {
    try {
      console.log('api: Initiating adaptive chat request', {
        message,
        filters,
        adaptiveSettings,
      });

      const response = await fetch(`${this.baseUrl}/chat/adaptive`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          message,
          filters,
          prioritize_earlier: adaptiveSettings.prioritize_earlier
        }),
        signal: options?.signal,
      });

      console.log('api: Received initial response', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
      });

      if (!response.ok) {
        console.error('api: Request failed', {
          status: response.status,
          statusText: response.statusText,
        });
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const parser = createParser((event: ParseEvent) => {
        console.log('api: Parsing SSE event', {
          type: event.type,
          data: event.type === 'event' ? event.data : undefined,
        });

        if (event.type === 'event' && event.data) {
          try {
            const data = JSON.parse(event.data) as AdaptiveSSEResponse;
            console.log('api: Parsed SSE data', {
              type: data.type,
              data: data,
            });
            
            if (data.type === 'reasoning_update' && callbacks.onReasoningUpdate) {
              const reasoningData = data as AdaptiveReasoningUpdate;
              console.log('api: Processing reasoning update', {
                steps: reasoningData.reasoning_steps,
              });
              callbacks.onReasoningUpdate?.(reasoningData.reasoning_steps);
            } else if (data.type === 'final_response' && callbacks.onFinalResponse) {
              const finalData = data as AdaptiveFinalResponse;
              console.log('api: Processing final response', {
                hasResponse: Boolean(finalData.response),
                hasChunkIds: Boolean(finalData.chunk_ids?.length),
                chunkIdsCount: finalData.chunk_ids?.length,
              });
              
              if (!finalData.response) {
                console.warn('api: Empty response in final_response event', {
                  data: finalData,
                });
                callbacks.onError?.('Received empty response from the server');
                return;
              }
              
              // Initialize empty arrays if not present
              if (!finalData.chunk_ids) finalData.chunk_ids = [];
              if (!finalData.document_ids) finalData.document_ids = [];
              
              // Call the callback with the response we have
              callbacks.onFinalResponse?.(finalData);
            } else if (data.type === 'error' && callbacks.onError) {
              const errorData = data as AdaptiveErrorResponse;
              console.error('api: Received error event', {
                error: errorData.error,
                details: errorData.details,
                fullData: errorData,
              });
              callbacks.onError?.(errorData.error);
            }
          } catch (e) {
            console.error('api: Failed to parse SSE event', {
              error: e,
              rawData: event.data,
            });
            callbacks.onError?.('Error processing server response');
          }
        }
      });

      const reader = response.body?.getReader();
      if (!reader) {
        console.error('api: Failed to get response reader');
        throw new Error('Response body cannot be read');
      }

      const decoder = new TextDecoder();
      console.log('api: Starting to read SSE stream');
      
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            console.log('api: SSE stream completed');
            break;
          }
          
          const chunk = decoder.decode(value, { stream: true });
          console.log('api: Received chunk', {
            chunkLength: chunk.length,
            chunk: chunk.substring(0, 100) + (chunk.length > 100 ? '...' : ''),
          });
          parser.feed(chunk);
        }
      } catch (e) {
        console.error('api: Error reading SSE stream', {
          error: e,
          errorMessage: e instanceof Error ? e.message : String(e),
        });
        callbacks.onError?.('Connection error while receiving updates');
        throw e;
      }
    } catch (e) {
      console.error('api: Fatal error in adaptive chat', {
        error: e,
        errorMessage: e instanceof Error ? e.message : String(e),
      });
      callbacks.onError?.(e instanceof Error ? e.message : 'Unknown error occurred');
      throw e;
    }
  }

  /**
   * Get random document chunks for testing
   */
  async getRandomChunks(count: number = 1): Promise<ChunkBatchResponse> {
    return this.fetch<ChunkBatchResponse>(`/random-chunks?count=${count}`, {
      method: 'GET',
    });
  }

  /**
   * Get random documents for testing
   */
  async getRandomDocuments(count: number = 1): Promise<DocumentBatchResponse> {
    return this.fetch<DocumentBatchResponse>(`/random-documents?count=${count}`, {
      method: 'GET',
    });
  }

  /**
   * Compress chunk IDs into a compact string for URL usage
   * This is a client-side approach to reduce URL size without backend changes
   */
  compressChunkIds(chunkIds: string[]): string {
    if (!chunkIds || chunkIds.length === 0) return '';
    
    try {
      // First, ensure all values are strings and properly formatted
      const sanitizedIds = chunkIds.map(id => {
        // Remove any potential unsafe characters but preserve digits
        return String(id).trim().replace(/[^\w\d-]/g, '');
      }).filter(id => id.length > 0);
      
      if (sanitizedIds.length === 0) {
        throw new Error('No valid chunk IDs after sanitization');
      }
      
      // Serialize as JSON
      const jsonString = JSON.stringify(sanitizedIds);
      
      // Use built-in compression if available
      if (typeof window !== 'undefined' && window.btoa) {
        // NOTE: We do NOT apply encodeURIComponent here - that will be done by the browser
        // when it puts the value in the URL. If we did it now, it would cause double-encoding.
        return window.btoa(jsonString);
      }
      
      // Fallback for environments without btoa
      return Buffer.from(jsonString).toString('base64');
    } catch (error) {
      console.error('Error compressing chunk IDs:', error);
      // Fallback to comma-separated format with sanitized IDs if compression fails
      return chunkIds
        .map(id => String(id).trim().replace(/[^\w\d-]/g, ''))
        .filter(id => id.length > 0)
        .join(',');
    }
  }

  /**
   * Decompress a compact string back into chunk IDs array
   */
  decompressChunkIds(compressed: string): string[] {
    if (!compressed) return [];
    
    try {
      // Check if it's a compressed string (try to decode)
      let jsonString: string;
      try {
        // Use built-in decompression if available
        if (typeof window !== 'undefined' && window.atob) {
          // Don't do decodeURIComponent again - the input should already be decoded
          jsonString = window.atob(compressed);
        } else {
          // Fallback for environments without atob
          jsonString = Buffer.from(compressed, 'base64').toString();
        }
        
        // Parse the JSON
        const parsedData = JSON.parse(jsonString);
        
        // Validate it's an array and ensure all values are safe strings
        if (Array.isArray(parsedData)) {
          // Log success for debugging
          console.log('Successfully decompressed IDs:', 
            parsedData.slice(0, 3).map(id => String(id)), 
            `(${parsedData.length} total)`);
            
          return parsedData
            .map(id => String(id).trim())
            .filter(id => id.length > 0);
        }
      } catch (e) {
        // Not a valid compressed string, try as comma-separated
        console.warn('Failed to decompress as Base64, trying as comma-separated:', e);
      }
      
      // Fallback: treat as comma-separated IDs
      return compressed
        .split(',')
        .map(id => id.trim())
        .filter(id => id.length > 0);
    } catch (error) {
      console.error('Error decompressing chunk IDs:', error);
      return [];
    }
  }
}

// Export a singleton instance
export default new ApiClient(); 