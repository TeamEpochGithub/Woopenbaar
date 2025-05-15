export interface PeriodFilter {
  startDate?: string | null;
  endDate?: string | null;
}

export interface FilterOptions {
  period?: PeriodFilter | null;
  docTypes?: string[] | null;
} 