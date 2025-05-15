import { useState, useRef, useEffect } from 'react';
import styles from './TypeFilter.module.css';

interface TypeFilterProps {
  onChange: (docTypes: string[]) => void;
}

export const TypeFilter: React.FC<TypeFilterProps> = ({ onChange }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [isActive, setIsActive] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const docTypes = [
    'Email',
    'PDF',
    'Word processing',
    'Presentation',
    'Spreadsheet',
    'Chatbericht',
  ]; // TODO: Fetch from API/local data
  
  useEffect(() => {
    // Close the dropdown when clicking outside
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  useEffect(() => {
    onChange(selectedTypes);
    setIsActive(selectedTypes.length > 0);
  }, [selectedTypes, onChange]);
  
  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };
  
  const handleTypeSelect = (type: string) => {
    setSelectedTypes(prev => {
      if (prev.includes(type)) {
        return prev.filter(t => t !== type);
      } else {
        return [...prev, type];
      }
    });
  };
  
  const clearFilter = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedTypes([]);
  };
  
  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        className={`${styles.filterButton} ${isActive ? styles.activeFilter : ''}`}
        onClick={toggleDropdown}
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        <div className="flex items-center flex-1 min-w-0">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-1 text-gray-700 shrink-0">
            <path fillRule="evenodd" d="M2 3.75A.75.75 0 0 1 2.75 3h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 3.75Zm0 4.167a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Zm0 4.166a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Zm0 4.167a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Z" clipRule="evenodd" />
          </svg>
          <span className="truncate">Document type</span>
        </div>
        {selectedTypes.length > 0 && (
          <span className={styles.countBadge}>
            {selectedTypes.length}
          </span>
        )}
      </button>
      
      {isOpen && (
        <div className={styles.dropdownMenu}>
          <div className="px-3 py-2">
            <h3 className={styles.menuHeader}>Selecteer document types</h3>
            <div className="space-y-1">
              {docTypes.map(type => (
                <div key={type} className={styles.checkboxItem}>
                  <input
                    type="checkbox"
                    id={`type-${type}`}
                    checked={selectedTypes.includes(type)}
                    onChange={() => handleTypeSelect(type)}
                    className={styles.checkboxInput}
                  />
                  <label htmlFor={`type-${type}`} className={styles.checkboxLabel}>
                    {type}
                  </label>
                </div>
              ))}
            </div>
          </div>
          <div className="border-t border-gray-100 mt-2 pt-2 px-3 flex justify-end">
            <button 
              className={styles.clearButton}
              onClick={clearFilter}
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}; 