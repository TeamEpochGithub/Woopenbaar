import { FilterOptions } from '@/types/filters';
import { AdaptiveSettings } from '@/types/adaptive';
import { StandardChatResponse, AdaptiveFinalResponse } from '@/types/api';
import apiClient from '@/lib/api';

interface AdaptiveCallbacks {
  onReasoningUpdate?: (reasoningSteps: string[]) => void;
  onFinalResponse?: (data: AdaptiveFinalResponse) => void;
  onError?: (error: string) => void;
}

export class ChatService {
  private isProcessing: boolean = false;
  private abortController: AbortController | null = null;

  async cancelCurrentRequest() {
    if (this.abortController) {
      this.abortController.abort();
      this.isProcessing = false;
      this.abortController = null;
    }
  }

  // Send a message using standard RAG
  async sendMessage(message: string, filters: FilterOptions, adaptiveSettings?: AdaptiveSettings): Promise<StandardChatResponse> {
    if (this.isProcessing) {
      throw new Error('Already processing a message');
    }
    
    this.abortController = new AbortController();
    this.isProcessing = true;
    
    try {
      console.log('ChatService: Sending standard message', {
        message,
        filters,
        prioritize_earlier: adaptiveSettings?.prioritize_earlier || false,
      });

      const data = await apiClient.sendChatMessage(
        message, 
        filters, 
        adaptiveSettings?.prioritize_earlier || false,
        { signal: this.abortController.signal }
      );

      console.log('ChatService: Received standard response', {
        hasResponse: Boolean(data.response),
        hasChunks: Boolean(data.chunk_ids?.length),
        chunksCount: data.chunk_ids?.length,
        hasChunkIds: Boolean(data.chunk_ids?.length),
        chunkIdsCount: data.chunk_ids?.length,
      });

      return data;
    } catch (error) {
      console.error('ChatService: Error in standard message', error);
      throw error;
    } finally {
      this.isProcessing = false;
      this.abortController = null;
    }
  }
  
  // Send a message using adaptive RAG with streaming
  async sendAdaptiveMessage(
    message: string, 
    filters: FilterOptions, 
    settings: AdaptiveSettings,
    callbacks: AdaptiveCallbacks = {}
  ): Promise<void> {
    if (this.isProcessing) {
      throw new Error('Already processing a message');
    }
    
    this.abortController = new AbortController();
    this.isProcessing = true;
    
    try {
      await apiClient.sendAdaptiveChatMessage(
        message, 
        filters, 
        settings,
        callbacks,
        { signal: this.abortController.signal }
      );
    } finally {
      this.isProcessing = false;
      this.abortController = null;
    }
  }
}

export default new ChatService(); 