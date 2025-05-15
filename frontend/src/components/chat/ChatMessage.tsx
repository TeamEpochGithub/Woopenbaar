import { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'marked-react';
import { ReasoningDisplay } from '@/components/adaptive/ReasoningDisplay';
import styles from './ChatMessage.module.css';
import { DocumentChunk, ChunkedDocument } from '@/types/api';
import apiClient from '@/lib/api';
import { useRouter } from 'next/router';

// Extend Window interface to allow our custom property
declare global {
  interface Window {
    sourceClickHandler?: (index: number | string) => void;
  }
}

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  chunks?: DocumentChunk[];
  documents?: ChunkedDocument[];
  chunk_ids?: number[];
  document_ids?: string[];
  reasoningSteps?: string[];
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  isUser,
  message,
  chunks = [],
  documents = [],
  chunk_ids = [],
  document_ids = [],
  reasoningSteps = [],
}) => {
  const [processedMessage, setProcessedMessage] = useState<string>(message);
  const componentMountedRef = useRef(false);
  const router = useRouter();
  
  useEffect(() => {
    componentMountedRef.current = true;
    
    return () => {
      componentMountedRef.current = false;
    };
  }, []);
  
  useEffect(() => {
    // Process chunk IDs for both adaptive and non-adaptive responses
    if (!isUser) {
      const hasChunks = chunks && chunks.length > 0;
      const hasChunkIds = chunk_ids && chunk_ids.length > 0;
      
      console.log('Processing message for citation links:', {
        hasChunks,
        chunksCount: chunks.length,
        hasChunkIds,
        chunkIdsCount: chunk_ids?.length || 0
      });
      
      // Use chunks if available, otherwise create placeholder chunks from chunk_ids
      let chunksToUse = chunks;
      
      if (!hasChunks && hasChunkIds) {
        // Create placeholder chunks from chunk_ids
        chunksToUse = chunk_ids.map((id, index) => ({
          uuid: id,
          content: `Source ${index + 1}`,
          type: 'text',
          link: '',
          parent_document: document_ids && document_ids[index] ? document_ids[index] : undefined
        }));
        console.log('Created placeholder chunks from chunk_ids:', chunksToUse);
      }
      
      if (chunksToUse.length > 0) {
        const processed = processChunkIds(message, chunksToUse);
        setProcessedMessage(processed);
      } else {
        setProcessedMessage(message);
      }
    } else {
      setProcessedMessage(message);
    }
  }, [message, chunks, chunk_ids, document_ids, isUser]);
  
  // Navigate to sources page with specific source highlighted
  const handleSourceClick = (sourceIndex: number | string) => {
    console.log('Source citation clicked:', sourceIndex);
    
    // Handle multi-citations (e.g. [1,2,4])
    if (typeof sourceIndex === 'string' && sourceIndex.includes(',')) {
      // This is a multi-citation, extract the individual IDs
      const individualIndices = sourceIndex.split(',').map(index => parseInt(index.trim(), 10));
      console.log('Multi-citation indices:', individualIndices);
      
      // Collect all chunk IDs for these indices
      let chunkIdsToUse: string[] = [];
      
      if (chunks?.length > 0) {
        // If we have chunks, map each index to its chunk
        individualIndices.forEach(index => {
          const arrayIndex = index - 1; // Convert 1-based to 0-based
          if (arrayIndex >= 0 && arrayIndex < chunks.length) {
            chunkIdsToUse.push(chunks[arrayIndex].uuid.toString());
          }
        });
      } else if (chunk_ids?.length > 0) {
        // If we have chunk_ids but not chunks, map each index to its chunk_id
        individualIndices.forEach(index => {
          const arrayIndex = index - 1; // Convert 1-based to 0-based
          if (arrayIndex >= 0 && arrayIndex < chunk_ids.length) {
            chunkIdsToUse.push(chunk_ids[arrayIndex].toString());
          }
        });
      }
      
      if (chunkIdsToUse.length === 0) {
        console.warn('No valid chunk IDs found for multi-citation indices:', individualIndices);
        return;
      }
      
      console.log('Opening sources with chunk IDs from multi-citation:', chunkIdsToUse);
      
      // Compress the chunk IDs and open in new tab
      const compressedIds = apiClient.compressChunkIds(chunkIdsToUse);
      const url = new URL(`/sources?ids=${compressedIds}`, window.location.origin);
      window.open(url.toString(), '_blank');
      return;
    }
    
    // Single citation handling (existing code)
    // In our system, citations are 1-indexed but arrays are 0-indexed
    // So we need to convert from the displayed index (1-based) to the array index (0-based)
    const arrayIndex = typeof sourceIndex === 'number' ? sourceIndex - 1 : parseInt(sourceIndex, 10) - 1;
    
    let sourceChunks = chunks;
    
    // If no chunks but we have chunk_ids, use those
    if ((!sourceChunks || sourceChunks.length === 0) && chunk_ids && chunk_ids.length > 0) {
      console.log('Using chunk_ids instead of chunks for navigation');
      
      if (arrayIndex < 0 || arrayIndex >= chunk_ids.length) {
        console.error('Invalid source index:', sourceIndex, '(array index:', arrayIndex, '), available chunk_ids:', chunk_ids.length);
        return;
      }
      
      // Navigate directly using the chunk_id
      const chunkId = chunk_ids[arrayIndex].toString();
      console.log('Navigating to source with chunk ID:', chunkId);
      
      // Compress single ID for URL
      const compressedId = apiClient.compressChunkIds([chunkId]);
      
      // Open in a new tab
      const url = new URL(`/sources?ids=${compressedId}`, window.location.origin);
      window.open(url.toString(), '_blank');
      return;
    }
    
    // Using actual chunks
    if (!sourceChunks || arrayIndex < 0 || arrayIndex >= sourceChunks.length) {
      console.error('Invalid source index:', sourceIndex, '(array index:', arrayIndex, '), available chunks:', sourceChunks?.length || 0);
      return;
    }
    
    const chunk = sourceChunks[arrayIndex];
    console.log('Navigating to source chunk:', {
      index: arrayIndex,
      uuid: chunk.uuid,
      content: chunk.content.substring(0, 50) + (chunk.content.length > 50 ? '...' : '')
    });
    
    // Compress single ID for URL
    const compressedId = apiClient.compressChunkIds([chunk.uuid.toString()]);
    
    // Open in a new tab
    const url = new URL(`/sources?ids=${compressedId}`, window.location.origin);
    window.open(url.toString(), '_blank');
  };
  
  // Process chunk IDs in the message text
  const processChunkIds = (content: string, chunks: DocumentChunk[]): string => {
    if (!chunks) return content;
    
    console.log('Processing citations in message', {
      contentLength: content.length,
      chunksLength: chunks.length,
      chunkIds: chunks.map(c => c.uuid).slice(0, 5) // First 5 chunk IDs
    });
    
    // Sanitize the content by escaping HTML special characters
    const sanitizeHtml = (html: string) => {
      return html
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    };
    
    // First sanitize the content
    let processedContent = sanitizeHtml(content);
    
    // Debug log what the message contains
    console.log('Message snippet:', processedContent.substring(0, 200) + (processedContent.length > 200 ? '...' : ''));
    
    // Check for single citation patterns in the message
    const singleCitationPattern = /\[(\d+)\]/g;
    const singleCitationMatches = processedContent.match(singleCitationPattern);
    
    // Check for multi-citation patterns in the message
    const multiCitationPattern = /\[(\d+(?:,\s*\d+)+)\]/g;
    const multiCitationMatches = processedContent.match(multiCitationPattern);
    
    console.log('Found single citation patterns:', singleCitationMatches || 'none');
    console.log('Found multi-citation patterns:', multiCitationMatches || 'none');
    
    // Process multi-citations first to avoid conflicts with single citations
    if (multiCitationMatches && multiCitationMatches.length > 0) {
      console.log('Multi-citations found:', multiCitationMatches.length, 'matches:', multiCitationMatches);
      
      processedContent = processedContent.replace(multiCitationPattern, (match, ids) => {
        const formattedIds = ids.replace(/\s+/g, ''); // Remove any whitespace between numbers
        
        console.log(`Found multi-citation: [${formattedIds}]`);
        
        // Check if the citation contains valid indices
        const individualIndices = formattedIds.split(',').map(id => parseInt(id, 10));
        const isValid = individualIndices.some(index => {
          const arrayIndex = index - 1;
          return arrayIndex >= 0 && arrayIndex < chunks.length;
        });
        
        return `<span class="${styles.sourceId}" data-source-id="${formattedIds}" onclick="window.sourceClickHandler('${formattedIds}')" title="Klik om deze bronnen te bekijken">
          <span class="${styles.sourceIdNumber}">[${formattedIds}]</span>
        </span>`;
      });
    }
    
    // Then process single citations
    if (singleCitationMatches && singleCitationMatches.length > 0) {
      console.log('Single citations found:', singleCitationMatches.length, 'matches:', singleCitationMatches);
      
      processedContent = processedContent.replace(singleCitationPattern, (match, id) => {
        const displayIndex = parseInt(id, 10);
        const arrayIndex = displayIndex - 1; // Convert to 0-based index for arrays
        
        console.log(`Found citation: [${displayIndex}], array index: ${arrayIndex}, valid: ${arrayIndex >= 0 && arrayIndex < chunks.length}`);
        
        if (arrayIndex >= 0 && arrayIndex < chunks.length) {
          // Since our citations are already 1-based (per the prompt), don't add 1 to the display
          return `<span class="${styles.sourceId}" data-source-id="${displayIndex}" onclick="window.sourceClickHandler(${displayIndex})" title="Klik om deze bron te bekijken">
            <span class="${styles.sourceIdNumber}">[${displayIndex}]</span>
          </span>`;
        }
        
        // If the index is out of range, keep it as is but still make it look clickable
        return `<span class="${styles.sourceId}" data-source-id="${displayIndex}" onclick="window.sourceClickHandler(${displayIndex})" title="Klik om deze bron te bekijken">
          <span class="${styles.sourceIdNumber}">[${displayIndex}]</span>
        </span>`;
      });
    }
    
    return processedContent;
  };
  
  // Count iterations from reasoning steps
  const countReasoningIterations = (reasoningSteps: string[]): number => {
    if (!reasoningSteps || reasoningSteps.length === 0) {
      return 0;
    }
    
    // Count steps that start with "Iteratie" or "Zoeken naar"
    const iterationSteps = reasoningSteps.filter(step => 
      step.startsWith('Iteratie') || step.startsWith('Zoeken naar:')
    );
    
    if (iterationSteps.length > 0) {
      return iterationSteps.length;
    }
    
    // Alternative: count all steps in multi-hop reasoning after initial strategy selection
    if (reasoningSteps.some(step => step.includes('meervoudige zoekopdracht'))) {
      // Subtract the strategy selection step
      return reasoningSteps.length - 1;
    }
    
    return 0;
  };
  
  // Open sources in a new tab
  const handleViewSources = () => {
    if ((!chunk_ids || chunk_ids.length === 0) && (!chunks || chunks.length === 0)) {
      console.warn('No chunk IDs or chunks available to view');
      return;
    }
    
    let chunkIdsToUse: string[] = [];
    
    // Use chunk_ids if available, otherwise extract IDs from chunks
    if (chunk_ids && chunk_ids.length > 0) {
      // Convert any numeric IDs to strings
      chunkIdsToUse = chunk_ids.map(id => id.toString());
    } else if (chunks && chunks.length > 0) {
      // Convert any numeric UUIDs to strings
      chunkIdsToUse = chunks.map(chunk => chunk.uuid.toString());
    }
    
    if (chunkIdsToUse.length === 0) {
      console.warn('No chunk IDs available after processing');
      return;
    }
    
    // Log the chunk IDs for debugging
    console.log('Opening sources with chunk IDs:', 
      chunkIdsToUse.slice(0, 3), 
      `(${chunkIdsToUse.length} total)`);
    
    // Compress the chunk IDs to reduce URL size
    const compressedIds = apiClient.compressChunkIds(chunkIdsToUse);
    
    // Use compressed IDs in the URL (no need to use encodeURIComponent, as
    // the URL constructor will handle that automatically)
    const url = new URL('/sources', window.location.origin);
    url.searchParams.append('ids', compressedIds);
    
    window.open(url.toString(), '_blank');
  };

  // Set up the global click handler
  useEffect(() => {
    // Only set up the handler if we have chunks or chunk_ids (i.e., there are citations to click)
    const hasChunks = chunks && chunks.length > 0;
    const hasChunkIds = chunk_ids && chunk_ids.length > 0;
    
    if (isUser || (!hasChunks && !hasChunkIds)) {
      return;
    }
    
    console.log('Setting up sourceClickHandler');
    
    // Create a global click handler that the inline onclick can call
    window.sourceClickHandler = (index: number | string) => {
      console.log('sourceClickHandler called with index:', index);
      handleSourceClick(index);
    };
    
    // Cleanup
    return () => {
      delete window.sourceClickHandler;
    };
  }, [chunks, chunk_ids, isUser, router]);
  
  return (
    <div className="flex flex-col w-full mb-8">
      {/* Reasoning section (only for bot messages) */}
      {!isUser && reasoningSteps.length > 0 && (
        <ReasoningDisplay steps={reasoningSteps} />
      )}
      
      {/* Search summary and source button (only for bot messages with sources) */}
      {!isUser && (chunks?.length > 0 || chunk_ids?.length > 0) && (
        <div className={`${styles.searchSummary} flex items-center justify-between mb-2`}>
          <div className="summary-text">
            {reasoningSteps.length === 0 && 'Enkelvoudige documentraadpleging'}
            {reasoningSteps.length > 0 && (
              <>
                Meervoudige zoekopdracht: {chunks?.length || chunk_ids?.length} {(chunks?.length || chunk_ids?.length) === 1 ? 'document' : 'documenten'} gebruikt
                {countReasoningIterations(reasoningSteps) > 0 && ` in ${countReasoningIterations(reasoningSteps)} ${countReasoningIterations(reasoningSteps) === 1 ? 'iteratie' : 'iteraties'}`}
              </>
            )}
          </div>
          
          <button
            onClick={handleViewSources}
            className={styles.viewSourcesBtn}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
              <path fillRule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
            </svg>
            Bekijk alle bronnen
          </button>
        </div>
      )}
      
      {/* Message content */}
      <div className={`${styles.message} ${isUser ? styles.userMessage : styles.botMessage}`}>
        {isUser ? (
          <span>{message}</span>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: processedMessage }} />
        )}
      </div>
    </div>
  );
};