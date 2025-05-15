import { useState, useEffect } from 'react';
import styles from './ReasoningDisplay.module.css';

interface ReasoningDisplayProps {
  steps: string[];
  isLive?: boolean;
}

export const ReasoningDisplay: React.FC<ReasoningDisplayProps> = ({ 
  steps, 
  isLive = false
}) => {
  const [showAllSteps, setShowAllSteps] = useState<boolean>(false);
  const [sourceCount, setSourceCount] = useState<number>(0);
  
  useEffect(() => {
    const newSourceCount = countSources(steps);
    setSourceCount(newSourceCount);
  }, [steps]);
  
  const countSources = (reasoningSteps: string[]): number => {
    let count = 0;
    for (const step of reasoningSteps) {
      if (step.includes('nieuwe relevante stukken informatie gevonden')) {
        const match = step.match(/(\d+)\s+nieuwe/);
        if (match) {
          count += parseInt(match[1]);
        }
      }
    }
    return count;
  };
  
  const toggleShowAllSteps = () => {
    setShowAllSteps(!showAllSteps);
  };
  
  if (steps.length === 0) {
    return null;
  }
  
  return (
    <div className={styles.reasoningContainer}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-blue-600">
              <path d="M12 .75a8.25 8.25 0 00-4.135 15.39c.686.398 1.115 1.008 1.134 1.623a.75.75 0 00.577.706c.352.083.71.148 1.074.195.323.041.6-.218.6-.544v-4.661a6.75 6.75 0 1110.5 0v4.661c0 .326.277.585.6.544.364-.047.722-.112 1.074-.195a.75.75 0 00.577-.706c.02-.615.448-1.225 1.134-1.623A8.25 8.25 0 0012 .75z" />
              <path fillRule="evenodd" d="M9.013 19.9a.75.75 0 01.877-.597 11.319 11.319 0 004.22 0 .75.75 0 11.28 1.473 12.819 12.819 0 01-4.78 0 .75.75 0 01-.597-.876zM9.754 22.344a.75.75 0 01.824-.668 13.682 13.682 0 002.844 0 .75.75 0 11.156 1.492 15.156 15.156 0 01-3.156 0 .75.75 0 01-.668-.824z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-sm font-medium text-blue-900">Zoekproces</span>
            {sourceCount > 0 && (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                {sourceCount} bronnen
              </span>
            )}
          </div>
        </div>
        
        {steps.length > 1 && (
          <button 
            className="text-blue-600 hover:text-blue-700 text-sm font-medium transition-colors"
            onClick={toggleShowAllSteps}
            aria-expanded={showAllSteps}
          >
            {showAllSteps ? 'Verberg details' : 'Toon details'}
          </button>
        )}
      </div>
      
      {!showAllSteps ? (
        // Collapsed view with truncated steps and connecting lines
        <div className={`${styles.reasoningStepsCollapsed} relative`}>
          {steps.map((step, index) => (
            <div 
              key={index}
              className={`${styles.reasoningStepCollapsed} text-sm text-blue-700 bg-white/60 p-2 rounded-md border border-blue-50`}
            >
              <p className="line-clamp-1">{step}</p>
            </div>
          ))}
        </div>
      ) : (
        // Expanded view with full content
        <div className={`mt-3 ${styles.reasoningStepsExpanded}`}>
          {steps.map((step, index) => (
            <div 
              key={index}
              className={`${styles.reasoningStepExpanded} text-sm text-blue-700 bg-white/60 p-3 rounded-md border border-blue-50`}
            >
              {step}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}; 