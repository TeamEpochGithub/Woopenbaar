"""Filter options for document retrieval."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


@dataclass
class PeriodFilter:
    """Date range filter for documents."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class FilterOptions:
    """Filter options for document retrieval.

    Attributes:
        period: Optional date range filter
        doc_types: Optional list of document types to include
    """

    period: Optional[PeriodFilter] = None
    doc_types: Optional[List[str]] = None


class FilterModel(BaseModel):
    """Filter options for retrieval requests."""

    period: Optional[Dict[str, str]] = None
    docTypes: Optional[List[str]] = None

    def to_filter_options(self) -> Optional[FilterOptions]:
        """Convert to FilterOptions object.

        Returns:
            Optional[FilterOptions]: Converted filter options
        """
        if not self.period and not self.docTypes:
            return None

        period = None
        if self.period:
            start_date = None
            end_date = None

            if self.period.get("startDate"):
                try:
                    start_date = datetime.fromisoformat(self.period["startDate"])
                except ValueError:
                    pass

            if self.period.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(self.period["endDate"])
                except ValueError:
                    pass

            if start_date or end_date:
                period = PeriodFilter(start_date=start_date, end_date=end_date)

        return FilterOptions(period=period, doc_types=self.docTypes)
