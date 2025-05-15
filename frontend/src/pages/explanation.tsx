import { useEffect, useState } from 'react';
import { Layout } from '@/components/Layout';
import MarkdownViewer from '@/components/MarkdownViewer';
import Head from 'next/head';

export default function ExplanationPage() {
  const [content, setContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function fetchMarkdownContent() {
      try {
        const response = await fetch('/explanation.md');
        if (!response.ok) {
          throw new Error('Failed to fetch explanation content');
        }
        const markdownContent = await response.text();
        setContent(markdownContent);
      } catch (error) {
        console.error('Error loading explanation content:', error);
        setContent('Could not load explanation content. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    }

    fetchMarkdownContent();
  }, []);

  return (
    <Layout>
      <Head>
        <title>Explanation | Woopenbaar</title>
        <meta name="description" content="Explanation of how Woopenbaar works" />
      </Head>
      
      <div className="max-w-4xl mx-auto py-8 px-4">
        <div className="mb-6 flex justify-between items-center">
          <h1 className="text-3xl font-medium text-gray-800">Over dit project</h1>
          <a href="/" className="text-primary hover:text-primary-dark text-sm flex items-center transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Terug naar chat
          </a>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-pulse text-gray-500">Loading explanation...</div>
          </div>
        ) : (
          <MarkdownViewer content={content} />
        )}
      </div>
    </Layout>
  );
} 