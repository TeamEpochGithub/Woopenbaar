import { useState, useEffect } from 'react';
import { AdaptiveSettings } from '@/types/adaptive';
import styles from './AdaptiveToggle.module.css';

interface AdaptiveToggleProps {
  onChange: (settings: AdaptiveSettings) => void;
}

export const AdaptiveToggle: React.FC<AdaptiveToggleProps> = ({ onChange }) => {
  const [useAdaptiveRAG, setUseAdaptiveRAG] = useState<boolean>(false);
  const [prioritizeEarlier, setPrioritizeEarlier] = useState<boolean>(false);
  
  useEffect(() => {
    // Notify parent component when settings change
    onChange({
      useAdaptiveRAG,
      prioritize_earlier: prioritizeEarlier
    });
  }, [useAdaptiveRAG, prioritizeEarlier, onChange]);
  
  const toggleAdaptiveMode = () => {
    setUseAdaptiveRAG(!useAdaptiveRAG);
  };
  
  const togglePrioritizeEarlier = () => {
    setPrioritizeEarlier(!prioritizeEarlier);
  };
  
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button 
        className={`${styles.toggleButton} ${useAdaptiveRAG ? styles.active : ''}`}
        onClick={toggleAdaptiveMode}
        aria-pressed={useAdaptiveRAG}
      >
        <div className="flex items-center flex-1">
          <span className="text-sm">EpochThink E5</span>
        </div>
        <div className={`${styles.toggleSwitch} ${useAdaptiveRAG ? styles.toggleSwitchActive : styles.toggleSwitchInactive}`}>
          <div className={`${styles.toggleKnob} ${useAdaptiveRAG ? styles.toggleKnobActive : styles.toggleKnobInactive}`}></div>
        </div>
      </button>
      
      <button 
        className={`${styles.toggleButton} ${prioritizeEarlier ? styles.active : ''}`}
        onClick={togglePrioritizeEarlier}
        aria-pressed={prioritizeEarlier}
      >
        <div className="flex items-center flex-1">
          <span className="text-sm">Prioritize Earlier</span>
        </div>
        <div className={`${styles.toggleSwitch} ${prioritizeEarlier ? styles.toggleSwitchActive : styles.toggleSwitchInactive}`}>
          <div className={`${styles.toggleKnob} ${prioritizeEarlier ? styles.toggleKnobActive : styles.toggleKnobInactive}`}></div>
        </div>
      </button>
    </div>
  );
}; 