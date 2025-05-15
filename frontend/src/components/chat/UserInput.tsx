import { useState, useRef } from 'react';

interface UserInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export const UserInput: React.FC<UserInputProps> = ({ onSend, disabled = false }) => {
  const [message, setMessage] = useState<string>('');
  const inputRef = useRef<HTMLInputElement>(null);
  
  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };
  
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  return (
    <div className="w-full flex gap-2 relative">
      <div className="w-full relative">
        <input
          type="text"
          ref={inputRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Typ hier uw vraag..."
          disabled={disabled}
          className="w-full py-3 px-4 pr-14 border border-gray-300 rounded-lg text-base shadow-sm focus:ring-2 focus:ring-primary/50 focus:border-primary disabled:bg-gray-100 disabled:cursor-not-allowed"
        />
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          className="absolute right-3 top-1/2 -translate-y-1/2 bg-primary hover:bg-primary-light px-4 py-1.5 rounded-md flex items-center justify-center cursor-pointer text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 mr-1">
            <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
          </svg>
          <span className="text-sm">Stuur</span>
        </button>
      </div>
    </div>
  );
}; 