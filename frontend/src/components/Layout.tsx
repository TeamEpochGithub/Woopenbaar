import Head from 'next/head';
import { ReactNode } from 'react';
import Link from 'next/link';
import Image from 'next/image';

interface LayoutProps {
  children: ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <>
      <Head>
        <title>Woopenbaar</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content="Woopenbaar - Een experimentele AI-chatinterface" />
      </Head>
      
      <div className="min-h-screen flex flex-col">
        <header className="bg-gradient-to-r from-primary-dark via-primary to-primary-light text-white py-3 shadow-md">
          <div className="max-w-4xl mx-auto px-4 flex items-center justify-between">
            <Link 
              href="/" 
              className="flex items-center"
              onClick={(e) => {
                if (window.location.pathname === '/') {
                  e.preventDefault();
                  window.location.reload();
                }
              }}
            >
              <Image
                src="/assets/logo_white.png"
                alt="Woopenbaar Logo"
                width={50}
                height={50}
                className="mr-3"
              />
              <h1 className="text-2xl font-medium tracking-tight">Woopenbaar</h1>
            </Link>
            <div className="flex items-center space-x-4">
              <div className="hidden md:block text-sm italic text-white/80">
                Experimentele WOO-document AI-chat
              </div>
              <Link 
                href="/explanation" 
                className="text-white/90 hover:text-white text-sm flex items-center transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
                </svg>
                Over dit project
              </Link>
            </div>
          </div>
        </header>
        
        <main className="flex-grow">
          {children}
        </main>
        
        <footer className="bg-primary-dark text-gray-400 py-2 mt-auto">
          <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
            <p className="text-xs italic">
              Dit is een experimentele AI-chatinterface. Antwoorden zijn mogelijk niet altijd accuraat.
            </p>
            <Link 
              href="/explanation" 
              className="text-xs text-gray-400 hover:text-white transition-colors"
            >
              Meer informatie
            </Link>
          </div>
        </footer>
      </div>
    </>
  );
}; 