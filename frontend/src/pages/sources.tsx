import { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import { DocumentChunk, ChunkedDocument } from '@/types/api';
import apiClient from '@/lib/api';
import { useRouter } from 'next/router';
import { Layout } from '@/components/Layout';
import { GetServerSideProps } from 'next';

type SourceType = {
  title: string;
  date?: string;
  content: string;
  link: string;
  index: number;
  vws_id?: string;
  uuid: string;
  parent_document?: string;
};

interface SourcesPageProps {
  initialSources: SourceType[];
  initialDocuments: ChunkedDocument[];
  initialError: string | null;
}

export const getServerSideProps: GetServerSideProps<SourcesPageProps> = async (context) => {
  try {
    const { chunk_ids, ids } = context.query;
    let chunkIdList: string[] = [];
    
    // Check if using compressed IDs format
    if (ids) {
      // Validate and decompress the IDs
      const compressedIds = Array.isArray(ids) ? ids[0] : ids;
      if (!compressedIds || typeof compressedIds !== 'string' || !compressedIds.trim()) {
        return {
          props: {
            initialSources: [],
            initialDocuments: [],
            initialError: 'Ongeldige ID-gegevens'
          }
        };
      }
      
      try {
        // Important: The server receives URL-encoded values, so we need to decode first
        const decodedValue = decodeURIComponent(compressedIds);
        console.log('Decoding compressed IDs:', decodedValue.substring(0, 30) + '...');
        
        // Decompress the IDs
        chunkIdList = apiClient.decompressChunkIds(decodedValue);
        
        if (chunkIdList.length === 0) {
          return {
            props: {
              initialSources: [],
              initialDocuments: [],
              initialError: 'Geen geldige bronfragmenten gevonden'
            }
          };
        }
      } catch (error) {
        console.error('Error decompressing chunk IDs:', error);
        return {
          props: {
            initialSources: [],
            initialDocuments: [],
            initialError: 'Er is een fout opgetreden bij het verwerken van de bronfragment-IDs'
          }
        };
      }
    } 
    // Legacy approach: direct chunk_ids parameter
    else if (chunk_ids) {
      // Parse IDs from query parameters
      const rawIds = Array.isArray(chunk_ids) ? chunk_ids[0] : chunk_ids;
      chunkIdList = rawIds.split(',').map(id => id.trim()).filter(id => id);
      
      if (chunkIdList.length === 0) {
        return {
          props: {
            initialSources: [],
            initialDocuments: [],
            initialError: 'Geen geldige bronfragmenten gevonden'
          }
        };
      }
    } else {
      return {
        props: {
          initialSources: [],
          initialDocuments: [],
          initialError: 'Geen bronfragmenten gevonden'
        }
      };
    }
    
    // Safety check: ensure all chunk IDs are valid
    // Make sure we're not passing the encoded string itself
    if (chunkIdList.some(id => id.length > 30 || id.includes('%'))) {
      console.error('Invalid chunk ID detected, possible encoding issue:', 
        chunkIdList.filter(id => id.length > 30 || id.includes('%')).slice(0, 2));
      return {
        props: {
          initialSources: [],
          initialDocuments: [],
          initialError: 'Ongeldig ID-formaat gedetecteerd'
        }
      };
    }
    
    // Now we have a valid chunkIdList, proceed with fetching chunks
    console.log(`Fetching ${chunkIdList.length} chunks from server...`, chunkIdList.slice(0, 3));
    const chunksResponse = await apiClient.getChunksBatch(chunkIdList);
    
    // Get document IDs from chunks to fetch full documents
    const documentIdSet = new Set<string>();
    chunksResponse.chunks.forEach(chunk => {
      if (chunk.parent_document) {
        documentIdSet.add(chunk.parent_document);
      }
    });
    
    let documentsData: ChunkedDocument[] = [];
    if (documentIdSet.size > 0) {
      // Fetch the full documents
      const documentIds = Array.from(documentIdSet);
      const documentsResponse = await apiClient.getDocumentsBatch(documentIds);
      documentsData = documentsResponse.documents;
    }
    
    // Create a map of document UUIDs to full documents for easier lookup
    const documentsMap = documentsData.reduce((acc, doc) => {
      acc[doc.uuid] = doc;
      return acc;
    }, {} as { [key: string]: ChunkedDocument });
    
    // Format sources for display
    const formattedSources = chunksResponse.chunks.map((chunk, index) => {
      const parentDocument = chunk.parent_document ? documentsMap[chunk.parent_document] : undefined;
      
      return {
        title: chunk.subject || `Fragment ${index + 1}`,
        date: chunk.content_date,
        content: chunk.content,
        link: chunk.link,
        index: index + 1,
        vws_id: parentDocument?.vws_id,
        uuid: chunk.uuid.toString(), // Convert to string to match SourceType
        parent_document: chunk.parent_document
      };
    });
    
    // Return both query params used and the results to help with debugging if needed
    return {
      props: {
        initialSources: formattedSources,
        initialDocuments: documentsData,
        initialError: null
      }
    };
    
  } catch (error) {
    console.error('Error fetching sources:', error);
    return {
      props: {
        initialSources: [],
        initialDocuments: [],
        initialError: 'Er is een fout opgetreden bij het ophalen van de bronnen'
      }
    };
  }
};

const SourcesPage: React.FC<SourcesPageProps> = ({ initialSources, initialDocuments, initialError }) => {
  const router = useRouter();
  const [sources, setSources] = useState<SourceType[]>(initialSources);
  const [documents, setDocuments] = useState<ChunkedDocument[]>(initialDocuments);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(initialError);
  const [selectedDocument, setSelectedDocument] = useState<ChunkedDocument | null>(null);
  const [highlightedChunk, setHighlightedChunk] = useState<DocumentChunk | null>(null);
  const [showDocumentModal, setShowDocumentModal] = useState<boolean>(false);
  const documentContentRef = useRef<HTMLDivElement>(null);
  const singleSourceRef = useRef<HTMLDivElement>(null);
  
  // Function to find a chunk in the document by its ID
  const findChunkById = (document: ChunkedDocument, chunkId: string): DocumentChunk | null => {
    // Try string comparison first (after converting numeric UUID to string if needed)
    let chunk = document.chunks.find(c => c.uuid.toString() === chunkId);
    
    // If not found, try with numeric comparison
    if (!chunk) {
      const numericId = parseInt(chunkId, 10);
      if (!isNaN(numericId)) {
        chunk = document.chunks.find(c => {
          // Compare as numbers if possible
          return typeof c.uuid === 'number' && c.uuid === numericId;
        });
      }
    }
    
    return chunk || null;
  };
  
  // Open document modal with highlighted chunk
  const viewFullDocument = (chunkId: string, documentId?: string) => {
    if (!documentId) {
      console.warn('No parent document ID available for this chunk');
      return;
    }
    
    // Find the document
    const document = documents.find(doc => doc.uuid === documentId);
    if (!document) {
      console.warn(`Document with ID ${documentId} not found`);
      return;
    }
    
    // Find the chunk either in the document chunks or in our sources
    let chunk = findChunkById(document, chunkId);
    
    // If still not found, create a chunk from the source
    if (!chunk) {
      const sourceWithChunk = sources.find(s => s.uuid === chunkId);
      if (sourceWithChunk) {
        // Try to convert the uuid to a number if possible
        let uuid: number;
        try {
          uuid = parseInt(sourceWithChunk.uuid, 10);
          if (isNaN(uuid)) {
            uuid = 0; // Fallback if parse fails
          }
        } catch {
          uuid = 0; // Fallback for any parsing issues
        }
        
        chunk = {
          uuid: uuid,
          content: sourceWithChunk.content,
          link: sourceWithChunk.link,
          type: '',
          parent_document: documentId
        };
      }
    }
    
    console.log('Found document:', document);
    console.log('Found chunk:', chunk);
    
    setSelectedDocument(document);
    setHighlightedChunk(chunk);
    setShowDocumentModal(true);
    
    // Scroll to highlighted chunk after modal is open
    setTimeout(() => {
      if (documentContentRef.current) {
        const mark = window.document.querySelector('.highlighted-chunk');
        if (mark) {
          mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }, 100);
  };
  
  // Process document content to highlight the selected chunk
  const processDocumentContent = (): string => {
    if (!selectedDocument || !highlightedChunk) return selectedDocument?.content || '';
    
    // Sanitize HTML to prevent XSS
    const sanitizeHtml = (html: string) => {
      return html
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    };
    
    const content = sanitizeHtml(selectedDocument.content);
    const highlightText = sanitizeHtml(highlightedChunk.content.trim());
    
    // First try exact match
    if (content.includes(highlightText)) {
      return content.replace(
        highlightText,
        `<span class="highlighted-chunk bg-yellow-200 p-1 rounded">${highlightText}</span>`
      );
    }
    
    // Try with relaxed whitespace matching
    const normalizedContent = content.replace(/\s+/g, ' ');
    const normalizedHighlight = highlightText.replace(/\s+/g, ' ');
    
    if (normalizedContent.includes(normalizedHighlight)) {
      // Find position in normalized content
      const startPos = normalizedContent.indexOf(normalizedHighlight);
      const endPos = startPos + normalizedHighlight.length;
      
      // Try to find corresponding position in original content
      let originalPos = 0;
      let normalizedPos = 0;
      let contentArray = content.split('');
      
      // Map normalized position to original position
      for (let i = 0; i < content.length; i++) {
        if (normalizedPos === startPos) {
          originalPos = i;
          break;
        }
        
        // Count non-whitespace or single space (not consecutive spaces)
        if (content[i] !== ' ' || (i > 0 && content[i-1] !== ' ')) {
          normalizedPos++;
        }
      }
      
      // Extract the segment from original content
      let chunkLength = 0;
      let chunkEndPos = originalPos;
      
      // Find the end position by counting characters that contribute to normalized length
      while (chunkLength < normalizedHighlight.length && chunkEndPos < content.length) {
        if (content[chunkEndPos] !== ' ' || (chunkEndPos > 0 && content[chunkEndPos-1] !== ' ')) {
          chunkLength++;
        }
        chunkEndPos++;
      }
      
      // Extract the actual text segment from the original content
      const actualText = content.substring(originalPos, chunkEndPos);
      
      // Replace the found segment with highlighted version
      return content.substring(0, originalPos) + 
             `<span class="highlighted-chunk bg-yellow-200 p-1 rounded">${actualText}</span>` + 
             content.substring(chunkEndPos);
    }
    
    // If still no match, add a warning
    return content + '\n\n<div class="mt-4 py-2 px-3 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-800">Het exacte fragment kon niet worden gevonden in het document. Mogelijk zijn er kleine tekstverschillen.</div>';
  };
  
  // If only one source is displayed, scroll to it and highlight it
  useEffect(() => {
    if (sources.length === 1 && singleSourceRef.current) {
      console.log('Single source detected, scrolling to and highlighting it');
      
      // Scroll to the single source and add a highlight effect
      singleSourceRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      singleSourceRef.current.classList.add('ring-2', 'ring-primary', 'ring-opacity-50');
      
      // Remove the highlight after a few seconds
      const timeout = setTimeout(() => {
        if (singleSourceRef.current) {
          singleSourceRef.current.classList.remove('ring-2', 'ring-primary', 'ring-opacity-50');
        }
      }, 3000);
      
      return () => clearTimeout(timeout);
    }
  }, [sources]);

  return (
    <>
      <Head>
        <title>
          {sources.length === 1 ? 'Bron' : 'Bronnen'} | Woopenbaar
        </title>
        <meta name="description" content="Bekijk de bronnen gebruikt voor het antwoord" />
      </Head>
      <Layout>
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto p-4 max-w-4xl">
          <header className="mb-8">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
              <h1 className="text-2xl font-semibold mb-2">
                {sources.length === 1 ? 'Bron' : 'Bronnen'}
              </h1>
              <p className="text-gray-600">
                {sources.length === 1 
                  ? 'Hieronder staat het geciteerde document dat is gebruikt in het antwoord.' 
                  : 'Hieronder staan alle documenten die zijn gebruikt om het antwoord te genereren.'}
              </p>
              
              {/* Document stats */}
              {!isLoading && !errorMessage && sources.length > 0 && (
                <div className="mt-4 bg-gray-50 rounded-lg p-3 border border-gray-100 text-sm text-gray-600">
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2 text-primary/70">
                      <path d="M10.75 16.82A7.462 7.462 0 0115 15.5c.71 0 1.396.098 2.046.282A.75.75 0 0018 15.06v-11a.75.75 0 00-.546-.721A9.006 9.006 0 0015 3a8.963 8.963 0 00-4.25 1.065V16.82zM9.25 4.065A8.963 8.963 0 005 3c-.85 0-1.673.118-2.454.339A.75.75 0 002 4.06v11c0 .332.22.624.546.721A7.488 7.488 0 005 15.5c1.036 0 2.024.174 2.922.483A6.869 6.869 0 019.25 16.82V4.065z" />
                    </svg>
                    <span><strong>{sources.length}</strong> {sources.length === 1 ? 'document' : 'documenten'} gevonden</span>
                  </div>
                </div>
              )}
            </div>
          </header>
          
          {isLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-pulse flex flex-col items-center">
                <div className="h-6 w-24 bg-gray-200 rounded mb-4"></div>
                <div className="h-32 w-full max-w-2xl bg-gray-200 rounded"></div>
                <div className="h-32 w-full max-w-2xl bg-gray-200 rounded mt-4"></div>
                <div className="h-32 w-full max-w-2xl bg-gray-200 rounded mt-4"></div>
              </div>
            </div>
          ) : errorMessage ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-800">
              <div className="flex items-center mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2 text-red-600">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                </svg>
                <h2 className="text-lg font-medium">Fout bij het laden van bronnen</h2>
              </div>
              <p>{errorMessage}</p>
            </div>
          ) : (
            <div className="space-y-6">
              {sources.map((source, index) => (
                <div 
                  key={index} 
                  className={`bg-white rounded-lg shadow-sm border border-gray-200 p-5 transition-all duration-300 ${sources.length === 1 ? 'ring-2 ring-primary ring-opacity-50' : ''}`}
                  ref={sources.length === 1 ? singleSourceRef : null}
                >
                  <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-3">
                    <div>
                      <h2 className="text-lg font-medium text-gray-800 flex items-center">
                        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-primary/10 text-primary text-sm mr-2">
                          {source.index}
                        </span>
                        {source.title}
                      </h2>
                      <div className="mt-1 space-y-1">
                        {source.date && (
                          <p className="text-gray-500 text-sm">
                            Datum: {source.date.split('T')[0]}
                          </p>
                        )}
                        {source.vws_id && (
                          <p className="text-gray-500 text-sm">
                            VWS ID: {source.vws_id}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="mt-2 sm:mt-0 flex flex-wrap gap-2">
                      {source.parent_document && (
                        <button
                          onClick={() => viewFullDocument(source.uuid, source.parent_document)}
                          className="text-primary hover:text-primary-dark text-sm flex items-center bg-primary/5 hover:bg-primary/10 px-3 py-1.5 rounded-md transition-colors"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
                            <path d="M3 3.5A1.5 1.5 0 014.5 2h6.879a1.5 1.5 0 011.06.44l4.122 4.12A1.5 1.5 0 0117 7.622V16.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 16.5v-13z" />
                          </svg>
                          Volledig document
                        </button>
                      )}
                      {source.link && (
                        <a
                          href={source.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:text-primary-dark text-sm flex items-center bg-primary/5 hover:bg-primary/10 px-3 py-1.5 rounded-md transition-colors"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
                            <path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" />
                            <path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z" clipRule="evenodd" />
                          </svg>
                          Bekijk origineel
                        </a>
                      )}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded border border-gray-100 text-gray-800 whitespace-pre-wrap text-sm overflow-auto max-h-96">
                    {source.content}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          <div className="mt-8 mb-6 flex justify-center">
        
          </div>
        </div>
      </div>
      
      {/* Full Document Modal */}
      {showDocumentModal && selectedDocument && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col">
            {/* Modal Header */}
            <div className="p-4 md:p-6 border-b border-gray-200">
              <div className="flex justify-between items-start mb-2">
                <h2 className="text-2xl font-semibold text-gray-800">
                  {selectedDocument.subject || 'Document'}
                </h2>
                <button 
                  onClick={() => setShowDocumentModal(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              {/* Document Metadata */}
              <div className="flex flex-wrap gap-4 text-sm text-gray-500">
                {selectedDocument.vws_id && (
                  <div className="flex items-center">
                    <span className="font-medium mr-1">VWS ID:</span> {selectedDocument.vws_id}
                  </div>
                )}
                
                {selectedDocument.create_date && (
                  <div className="flex items-center">
                    <span className="font-medium mr-1">Datum:</span> {selectedDocument.create_date}
                  </div>
                )}
                
                {selectedDocument.type && (
                  <div className="flex items-center">
                    <span className="font-medium mr-1">Type:</span> {selectedDocument.type}
                  </div>
                )}
              </div>
              
              {/* Document Links */}
              {selectedDocument.link && (
                <div className="mt-4">
                  <a
                    href={selectedDocument.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary-dark text-sm flex items-center bg-primary/5 hover:bg-primary/10 px-3 py-1.5 rounded-md transition-colors inline-flex"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
                      <path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" />
                      <path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z" clipRule="evenodd" />
                    </svg>
                    Bekijk origineel
                  </a>
                </div>
              )}
              
              {/* Highlighted content notice */}
              {highlightedChunk && (
                <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm">
                  <div className="flex items-center text-yellow-800">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2 text-yellow-600">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zm-1 9a1 1 0 100-2 1 1 0 000 2zm0-3a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    <span>De geel gemarkeerde tekst bevat het fragment dat is gebruikt in het antwoord.</span>
                  </div>
                </div>
              )}
            </div>
            
            {/* Modal Content */}
            <div className="p-4 md:p-6 flex-1 overflow-y-auto" ref={documentContentRef}>
              <div 
                className="bg-gray-50 rounded border border-gray-100 p-4 text-gray-800 whitespace-pre-wrap text-sm leading-relaxed"
                dangerouslySetInnerHTML={{ __html: processDocumentContent() }}
              />
            </div>
            
            {/* Modal Footer */}
            <div className="p-4 md:p-6 border-t border-gray-200 flex justify-between">
              {highlightedChunk && (
                <button
                  onClick={() => {
                    const mark = window.document.querySelector('.highlighted-chunk');
                    if (mark) {
                      mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                  }}
                  className="text-primary hover:text-primary-dark text-sm flex items-center bg-primary/5 hover:bg-primary/10 px-3 py-1.5 rounded-md transition-colors"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
                    <path fillRule="evenodd" d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z" clipRule="evenodd" />
                  </svg>
                  Ga naar gemarkeerd fragment
                </button>
              )}
              
              <button
                onClick={() => setShowDocumentModal(false)}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium ml-auto"
              >
                Sluiten
              </button>
            </div>
          </div>
        </div>
      )}
      </Layout>
    </>
  );
};

export default SourcesPage; 