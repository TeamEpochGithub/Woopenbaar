import { useState, useCallback } from 'react';
import { PeriodFilter } from './PeriodFilter';
import { TypeFilter } from './TypeFilter';
import { FilterOptions, PeriodFilter as PeriodFilterType } from '@/types/filters';
import styles from './FilterSystem.module.css';

interface FilterSystemProps {
  onChange: (filters: FilterOptions) => void;
}

export const FilterSystem: React.FC<FilterSystemProps> = ({ onChange }) => {
  const [periodFilter, setPeriodFilter] = useState<PeriodFilterType | null>(null);
  const [docTypes, setDocTypes] = useState<string[]>([]);
  
  const handlePeriodChange = useCallback((filter: PeriodFilterType) => {
    setPeriodFilter(filter);
    
    // Notify parent component of changes
    onChange({
      period: filter.startDate && filter.endDate ? filter : null,
      docTypes: docTypes.length > 0 ? docTypes : null
    });
  }, [docTypes, onChange]);
  
  const handleTypeChange = useCallback((types: string[]) => {
    setDocTypes(types);
    
    // Notify parent component of changes
    onChange({
      period: periodFilter?.startDate && periodFilter?.endDate ? periodFilter : null,
      docTypes: types.length > 0 ? types : null
    });
  }, [periodFilter, onChange]);
  
  return (
    <div className="flex flex-wrap items-center gap-2">
      <PeriodFilter onChange={handlePeriodChange} />
      <TypeFilter onChange={handleTypeChange} />
    </div>
  );
}; 