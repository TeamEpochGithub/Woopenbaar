import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { UserInput } from './UserInput';
import { FilterSystem } from '../filters/FilterSystem';
import { AdaptiveToggle } from '../adaptive/AdaptiveToggle';
import { ReasoningDisplay } from '../adaptive/ReasoningDisplay';
import ChatService from '@/services/ChatService';
import { ChatMessageData } from '@/types/chat';
import { FilterOptions } from '@/types/filters';
import { AdaptiveSettings } from '@/types/adaptive';
import Link from 'next/link';


export const ChatWindow: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [filters, setFilters] = useState<FilterOptions>({});
  const [adaptiveSettings, setAdaptiveSettings] = useState<AdaptiveSettings>({
    useAdaptiveRAG: false,
    prioritize_earlier: false
  });
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isLoadingChunks, setIsLoadingChunks] = useState<boolean>(false);
  const [chatActive, setChatActive] = useState<boolean>(false);
  const [liveReasoningSteps, setLiveReasoningSteps] = useState<string[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, liveReasoningSteps]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const handleSendMessage = async (message: string) => {
    console.log('Sending message:', message);
    console.log('Current filters:', filters);
    console.log('Adaptive settings:', adaptiveSettings);
    
    // Add user message to chat
    addMessage({
      message,
      isUser: true
    });
    
    // Show that chat is active
    if (!chatActive) {
      setChatActive(true);
    }
    
    setIsProcessing(true);
    setLiveReasoningSteps([]);
    
    try {
      if (adaptiveSettings.useAdaptiveRAG) {
        // Handle adaptive RAG mode with streaming
        console.log('Starting adaptive RAG request');
        setIsLoadingChunks(true);
        await ChatService.sendAdaptiveMessage(
          message,
          filters,
          adaptiveSettings,
          {
            onReasoningUpdate: (steps) => {
              console.log('Received reasoning update:', steps);
              setLiveReasoningSteps(steps);
              
              setTimeout(() => {
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              }, 100);
            },
            onFinalResponse: (data) => {
              console.log('Received final adaptive response:', data);
              setIsLoadingChunks(false);
              // Process the final response
              addMessage({
                message: data.response,
                isUser: false,
                chunks: data.chunks || [],
                documents: data.documents || [],
                chunk_ids: data.chunk_ids,
                document_ids: data.document_ids,
                reasoning: data.reasoning_steps,
                timestamp: data.timestamp
              });
              setLiveReasoningSteps([]);
            },
            onError: (error) => {
              console.error('Error from adaptive RAG:', error);
              setIsLoadingChunks(false);
              addMessage({
                message: `Er is een fout opgetreden: ${error}`,
                isUser: false
              });
              setLiveReasoningSteps([]);
            }
          }
        );
      } else {
        // Handle standard RAG
        console.log('Starting standard RAG request');
        setIsLoadingChunks(true);
        const data = await ChatService.sendMessage(message, filters, adaptiveSettings);
        console.log('Received standard response:', data);
        setIsLoadingChunks(false);
        
        // Add bot response to chat
        addMessage({
          message: data.response,
          isUser: false,
          chunks: data.chunks || [],
          documents: data.documents || [],
          chunk_ids: data.chunk_ids,
          document_ids: data.document_ids,
          reasoning: data.reasoning_steps,
          timestamp: data.timestamp
        });
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoadingChunks(false);
      addMessage({
        message: `Er is een fout opgetreden: ${error instanceof Error ? error.message : 'Onbekende fout'}`,
        isUser: false
      });
    } finally {
      setIsProcessing(false);
    }
  };
  
  const addMessage = (messageData: ChatMessageData) => {
    console.log('Adding message to chat:', messageData);
    setMessages(prev => [...prev, messageData]);
  };
  
  // Sample questions that can be used as conversation starters
  const sampleQuestions = [
    "Welke maatregelen werden genomen tijdens de coronacrisis?",
    "Hoe werd er gecommuniceerd met het publiek tijdens de crisis?",
    "Hoe verliep de besluitvorming rondom vaccinaties?",
    "Welke partijen waren betrokken bij de inkoop van beschermingsmiddelen?"
  ];
  
  // Function to handle clicking a sample question
  const handleSampleQuestionClick = (question: string) => {
    handleSendMessage(question);
  };
  
  return (
    <div className="flex flex-col items-center w-full h-full max-w-4xl mx-auto px-4">
      {/* Chat area - Takes full width and height */}
      <div className="w-full bg-white rounded-lg shadow-sm border border-gray-200 p-4 flex flex-col flex-grow h-[calc(100vh-180px)] mt-4">
        {!chatActive ? (
          <div className="flex-grow flex flex-col items-center justify-center">
            <div className="text-center max-w-xl w-full px-4 py-6 rounded-xl bg-gradient-to-br from-gray-50 to-white shadow-md border border-gray-100">
              <div className="mb-3 flex justify-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 text-primary">
                    <path d="M4.913 2.658c2.075-.27 4.19-.408 6.337-.408 2.147 0 4.262.139 6.337.408 1.922.25 3.291 1.861 3.405 3.727a4.403 4.403 0 0 0-1.032-.211 50.89 50.89 0 0 0-8.42 0c-2.358.196-4.04 2.19-4.04 4.434v4.286a4.47 4.47 0 0 0 2.433 3.984L7.28 21.53A.75.75 0 0 1 6 21v-4.03a48.527 48.527 0 0 1-1.087-.128C2.905 16.58 1.5 14.833 1.5 12.862V6.638c0-1.97 1.405-3.718 3.413-3.979Z" />
                    <path d="M15.75 7.5c-1.376 0-2.739.057-4.086.169C10.124 7.797 9 9.103 9 10.609v4.285c0 1.507 1.128 2.814 2.67 2.94 1.243.102 2.5.157 3.768.165l2.782 2.781a.75.75 0 0 0 1.28-.53v-2.39l.33-.026c1.542-.125 2.67-1.433 2.67-2.94v-4.286c0-1.505-1.125-2.811-2.664-2.94A49.392 49.392 0 0 0 15.75 7.5Z" />
                  </svg>
                </div>
              </div>
              
              <h2 className="text-xl font-medium text-gray-800 mb-3">Welkom bij Woopenbaar</h2>
              <p className="text-gray-600 mb-4 text-sm">Stel een vraag over de WOO-documenten of kies een van de voorbeeldvragen.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-md mx-auto">
                {sampleQuestions.map((question, index) => (
                  <button 
                    key={index}
                    onClick={() => handleSampleQuestionClick(question)}
                    className="text-left px-3 py-2 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg text-gray-700 transition-colors text-sm duration-200 flex items-center group"
                  >
                    <span className="mr-2 text-primary opacity-70 group-hover:opacity-100">•</span>
                    {question}
                  </button>
                ))}
              </div>
              
              <div className="mt-4 text-center">
                <div className="inline-flex items-center text-gray-500 text-xs">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3 h-3 mr-1 text-gray-400">
                    <path fillRule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM9 9a.75.75 0 0 0 0 1.5h.253a.25.25 0 0 1 .244.304l-.459 2.066A1.75 1.75 0 0 0 10.747 15H11a.75.75 0 0 0 0-1.5h-.253a.25.25 0 0 1-.244-.304l.459-2.066A1.75 1.75 0 0 0 9.253 9H9Z" clipRule="evenodd" />
                  </svg>
                  Antwoorden worden gegenereerd op basis van WOO-documenten
                </div>
                <Link href="/explanation" className="block mt-2 text-primary text-xs hover:underline">
                  Lees meer over hoe dit project werkt
                </Link>
              </div>
            </div>
          </div>
        ) : (
          <div 
            ref={messagesContainerRef} 
            className="w-full overflow-y-auto flex-grow mb-4"
          >
            {messages.map((msg, index) => (
              <ChatMessage
                key={index}
                isUser={msg.isUser}
                message={msg.message}
                chunks={msg.chunks}
                documents={msg.documents}
                chunk_ids={msg.chunk_ids}
                document_ids={msg.document_ids}
                reasoningSteps={msg.reasoning}
              />
            ))}
            
            {/* Live reasoning display during processing */}
            {isProcessing && adaptiveSettings.useAdaptiveRAG && liveReasoningSteps.length > 0 && (
              <ReasoningDisplay 
                steps={liveReasoningSteps} 
                isLive={true}
              />
            )}
            
            {/* Loading indicator for chunks fetching */}
            {isLoadingChunks && !isProcessing && (
              <div className="text-center py-2 text-gray-500 text-sm">
                <span className="inline-block animate-pulse">Documentinformatie ophalen...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
        
        {/* User input - Full width at the bottom */}
        <div className="w-full">
          <UserInput onSend={handleSendMessage} disabled={isProcessing} />
        </div>
        
        {/* Filters and settings - Now below the chat input */}
        <div className="w-full mt-3 flex justify-center">
          <div className="flex flex-wrap items-center gap-3 bg-gray-50 p-2 rounded-lg border border-gray-100 shadow-sm">
            <FilterSystem onChange={setFilters} />
            <AdaptiveToggle onChange={setAdaptiveSettings} />
          </div>
        </div>
      </div>
    </div>
  );
}; 