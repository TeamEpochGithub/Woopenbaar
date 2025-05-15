import { useState, useRef, useEffect } from 'react';
import { PeriodFilter as PeriodFilterType } from '@/types/filters';
import styles from './PeriodFilter.module.css';

interface PeriodFilterProps {
  onChange: (filter: PeriodFilterType) => void;
}

export const PeriodFilter: React.FC<PeriodFilterProps> = ({ onChange }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [isActive, setIsActive] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
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
    const filter: PeriodFilterType = {};
    
    if (startDate) {
      filter.startDate = startDate;
    }
    
    if (endDate) {
      filter.endDate = endDate;
    }
    
    onChange(filter);
    setIsActive(startDate !== '' || endDate !== '');
  }, [startDate, endDate, onChange]);
  
  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };
  
  const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setStartDate(e.target.value);
  };
  
  const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEndDate(e.target.value);
  };
  
  const clearFilter = (e: React.MouseEvent) => {
    e.stopPropagation();
    setStartDate('');
    setEndDate('');
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
            <path fillRule="evenodd" d="M5.75 2a.75.75 0 0 1 .75.75V4h7V2.75a.75.75 0 0 1 1.5 0V4h.25A2.75 2.75 0 0 1 18 6.75v8.5A2.75 2.75 0 0 1 15.25 18H4.75A2.75 2.75 0 0 1 2 15.25v-8.5A2.75 2.75 0 0 1 4.75 4H5V2.75A.75.75 0 0 1 5.75 2Zm-1 5.5c-.69 0-1.25.56-1.25 1.25v6.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-6.5c0-.69-.56-1.25-1.25-1.25H4.75Z" clipRule="evenodd" />
          </svg>
          <span className="truncate">Periode</span>
        </div>
        {isActive && (
          <span className="ml-auto bg-primary/10 text-primary rounded-full w-5 h-5 flex items-center justify-center text-xs font-medium shrink-0">
            ✓
          </span>
        )}
      </button>
      
      {isOpen && (
        <div className={styles.dropdownMenu}>
          <div className="px-3 py-2">
            <h3 className={styles.menuHeader}>Selecteer periode</h3>
            <div className="space-y-3">
              <div>
                <label htmlFor="start-date" className={styles.labelText}>
                  Van
                </label>
                <input
                  type="date"
                  id="start-date"
                  value={startDate}
                  onChange={handleStartDateChange}
                  className={styles.dateField}
                />
              </div>
              <div>
                <label htmlFor="end-date" className={styles.labelText}>
                  Tot
                </label>
                <input
                  type="date"
                  id="end-date"
                  value={endDate}
                  onChange={handleEndDateChange}
                  className={styles.dateField}
                />
              </div>
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